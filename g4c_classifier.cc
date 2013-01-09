#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
//#include <cuda.h>
#include <vector>
#include <map>
#include <set>

#include "g4c_classifier.h"

using namespace std;

static int verbose_level = 1;


#define PORT_BITS 9
#define PORT_MASK (((uint16_t)0xffff)>>(16-PORT_BITS))
#define get_eport(_p) ((int)((_p)&PORT_MASK))

#define PROTO_BITS 7
#define PROTO_MASK (((uint16_t)0xffff)>>(16-PROTO_BITS))
#define get_eproto(_p) ((int)((_p)&PROTO_MASK))

class State {
public:
    int id;
    bool fail;
    bool failon;
    map<int, int> go;
    set<int> outputs;
    bool final;
    int level;
    State(int sid=0, int slevel=0) : id(sid), fail(false), final(false), failon(false) {
	level = slevel;}    
};

class Classifier {
public:
    vector<State*> states;
    set<State*> *last_layer;
};

static int
_build_goto(g4c_pattern_t *ptns, int n, Classifier *cl)
{
    int sid = 1;
    int i;
    vector<State*> &states = cl->states;

    states.reserve(n<<3);

    State* s = new State(0, 0);
    states.push_back(s);

    // Src addr:
    for (i=0; i<n; i++) {
	g4c_pattern_t *ptn = ptns+i;
	int nsid = 0;

	s = states[0];
	for (int b=0; b<4; b++) {
	    if ((ptn->nr_src_netbits>>3) > b) // normal
	    {
		int v = (ptn->src_addr>>(b*8)) & 0xff;
		map<int, int>::iterator ite = s->go.find(v);
		if (ite != s->go.end()) { // go through existing path
		    nsid = ite->second;
		    s = states[nsid];
		} else { // need a new path
		    State *ns = new State(sid++, b+1);
		    states.push_back(ns);
		    s->go.insert(pair<int, int>(v, ns->id));
		    s = ns;				 
		}
	    }
	    else if ((ptn->nr_src_netbits>>3) == b) // fail now
	    {
		s->outputs.insert(ptn->idx);
		s->failon = true;
		break;
	    } else {
		fprintf(stderr, "src ip bug: b: %d, nr_src_netbits: %d\n",
			b, ptn->nr_src_netbits);
		assert(0);
	    }
	}
	if (ptn->nr_src_netbits == 32)
	    s->outputs.insert(ptn->idx);
    }

    State *ipa_0_out_fs = new State[11]; // shared fail nodes with 0 outputs
    for (int l=0; l<11; l++) {
	ipa_0_out_fs[l].id = sid++;
	ipa_0_out_fs[l].fail = true;
	ipa_0_out_fs[l].level = l+1;
	states.push_back(ipa_0_out_fs+l);

	if (l < 7) {
	    for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		ipa_0_out_fs[l].go.insert(pair<int, int>(v, sid));
	    }
	} else if (l < 9) {
	    for (int v=0; v<=PORT_MASK; v++)
		ipa_0_out_fs[l].go.insert(pair<int, int>(v, sid));
	} else if (l < 10) {
	    for (int v=0; v<=PROTO_MASK; v++)
		ipa_0_out_fs[l].go.insert(pair<int, int>(v, sid));
	}
    }

    s = states[0];
    set<State*> *layer = new set<State*>(), *next_layer = new set<State*>(), *tmplp;
    assert(layer && next_layer);
    layer->insert(s);
    for (int l=0; l<4; l++) // breadth first iteration
    {
	set<State*>::iterator ite;
	for (ite = layer->begin(); ite != layer->end(); ++ite) // go through current layer
	{
	    State* curs = *ite;
	    if (!curs->fail) // normal node
	    {
		map<int, int>::iterator goite = curs->go.begin();
		while(goite != curs->go.end()) {
		    State *ns = states[goite->second];
		    ns->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		    ++goite;
		    next_layer->insert(ns);
		}
	    }

	    if (curs->go.size() < G4C_IPA_STATE_SIZE) // build failure transitions
	    {
		State *fs;

		if (curs->outputs.size() == 0) {
		    fs = ipa_0_out_fs+l;
		} else {
		    fs = new State(sid++, l+1);
		    states.push_back(fs);
		    fs->fail = true;
		    fs->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		}
		
		for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		    if (curs->go.find(v) == curs->go.end()) // failure value
		    {
			curs->go.insert(pair<int, int>(v, fs->id));
		    }
		}
		next_layer->insert(fs);
	    }	    
	}

	tmplp = layer;
	layer = next_layer;
	next_layer = tmplp;
	next_layer->clear();
    }

    if (verbose_level > 0) {
	printf("INFO:: G4C Classifier: build_goto: has %lu src finals, %lu src nodes\n",
	       layer->size(), states.size());
    }

    // Dst addr:
    set<State*>::iterator site = layer->begin();
    while (site != layer->end())
    {
	s = *site;
	State* dstroot = s;
	int nsid = 0;

	set<int> ignore_ptns;

	set<int>::iterator oite = dstroot->outputs.begin();
	while (oite != dstroot->outputs.end()) {
	    g4c_pattern_t *ptn = ptns+(*oite);
	    for (int b=0; b<4; b++) // go through bytes
	    {
		if ((ptn->nr_dst_netbits>>3) > b) // normal
		{
		    int v = (ptn->dst_addr>>(b*8)) & 0xff;
		    map<int, int>::iterator pos = s->go.find(v);
		    if (pos != s->go.end()) {
			nsid = pos->second;
			s = states[nsid];
		    } else {
			State *ns = new State(sid++, b+5);
			states.push_back(ns);
			s->go.insert(pair<int, int>(v, ns->id));
			s = ns;
		    }
		}
		else if ((ptn->nr_dst_netbits>>3) == b) // fail now
		{
		    s->failon = true;
		    if (s != dstroot)
			s->outputs.insert(ptn->idx);
		    else
			ignore_ptns.insert(ptn->idx);
		    break;
		} else {
		    fprintf(stderr, "dst ip bug: b %d, nr_dst_netbits: %d\n",
			    b, ptn->nr_dst_netbits);
		    assert(0);
		}
	    } // bytes
	    if (ptn->nr_dst_netbits == 32)
		s->outputs.insert(ptn->idx);
	    s = dstroot;
	    ++oite;
	}

	map<int, int>::iterator goite = dstroot->go.begin();
	while (goite != dstroot->go.end()) {
	    State *ns = states[goite->second];
	    if (ignore_ptns.size() > 0) {
		ns->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    ++goite;
	    next_layer->insert(ns);
	    if (dstroot == ipa_0_out_fs+3)
		break;
	}
	
	if (dstroot != ipa_0_out_fs+3) // failure for dst root
	{
	    State *fs;

	    if (ignore_ptns.size() == 0) {
		fs = ipa_0_out_fs+4;
	    } else {
		fs = new State(sid++, 4+1);
		states.push_back(fs);
		fs->fail = true;
		fs->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }
	    
	    for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		if (dstroot->go.find(v) == dstroot->go.end()) // failure value
		{
		    dstroot->go.insert(pair<int, int>(v, fs->id));
		}
	    }

	    next_layer->insert(fs);
	}
	
	++site;
    } // dst sub tries goto done.

    tmplp = layer;
    layer = next_layer;
    next_layer = tmplp;
    next_layer->clear();

    // Build failure for dst:
    for (int l=5; l<8; l++) { // BFS on dst tries
	set<State*>::iterator ite;
	for (ite = layer->begin(); ite != layer->end(); ++ite) // current layer
	{
	    State* curs = *ite;
	    if (!curs->fail)
	    {
		map<int, int>::iterator goite = curs->go.begin();
		while(goite != curs->go.end()) {
		    State *ns = states[goite->second];
		    ns->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		    ++goite;
		    next_layer->insert(ns);
		}
	    }

	    if (curs->go.size() < G4C_IPA_STATE_SIZE)
	    {
		State *fs;

		if (curs->outputs.size() == 0) {
		    fs = ipa_0_out_fs+l;
		} else {
		    fs = new State(sid++, l+1);
		    states.push_back(fs);
		    fs->fail = true;
		    fs->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		}
		
		for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		    if (curs->go.find(v) == curs->go.end()) // failure value
		    {
			curs->go.insert(pair<int, int>(v, fs->id));
		    }
		}
		next_layer->insert(fs);
	    }
	}

	tmplp = layer;
	layer = next_layer;
	next_layer = tmplp;
	next_layer->clear();
    }

    if (verbose_level > 0) {
	printf("INFO:: G4C Classifier: build_goto: has %lu src+dst finals, %lu src+dst nodes\n",
	       layer->size(), states.size());
    }

    // Src port:
    site = layer->begin();
    while (site != layer->end())
    {
	s = *site;
	int nsid = 0;
	set<int> ignore_ptns;

	set<int>::iterator oite = s->outputs.begin();
	while (oite != s->outputs.end())
	{
	    g4c_pattern_t *ptn = ptns+(*oite);
	    if (ptn->src_port < 0 || ptn->src_port > PORT_MASK) {
		ignore_ptns.insert(ptn->idx);
	    } else {
		int v = get_eport(ptn->src_port);
		map<int, int>::iterator pos = s->go.find(v);
		if (pos != s->go.end()) {
		    nsid = pos->second;
		    states[nsid]->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 9); // level 9
		    states.push_back(ns);
		    s->go.insert(pair<int, int>(v, ns->id));
		    ns->outputs.insert(ptn->idx);
		}
	    }

	    ++oite;
	}

	map<int, int>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = states[goite->second];
	    if (ignore_ptns.size() > 0) {
		ns->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    ++goite;
	    next_layer->insert(ns);
	    if (s == ipa_0_out_fs+7)
		break;
	}

	if (s != ipa_0_out_fs+7)
	{
	    State *fs;

	    if (ignore_ptns.size() == 0)
		fs = ipa_0_out_fs+8;
	    else {
		fs = new State(sid++, 8+1);
		states.push_back(fs);
		fs->fail = true;
		fs->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    for (int v=0; v<= PORT_MASK; v++) {
		if (s->go.find(v) == s->go.end())
		    s->go.insert(pair<int, int>(v, fs->id));
	    }

	    next_layer->insert(fs);
	}
	
	++site;
    }

    tmplp = layer;
    layer = next_layer;
    next_layer = tmplp;
    next_layer->clear();

    // Dst port:
    site = layer->begin();
    while (site != layer->end())
    {
	s = *site;
	int nsid = 0;
	set<int> ignore_ptns;

	set<int>::iterator oite = s->outputs.begin();
	while (oite != s->outputs.end())
	{
	    g4c_pattern_t *ptn = ptns+(*oite);
	    if (ptn->dst_port < 0 || ptn->dst_port > PORT_MASK) {
		ignore_ptns.insert(ptn->idx);
	    } else {
		int v = get_eport(ptn->dst_port);
		map<int, int>::iterator pos = s->go.find(v);
		if (pos != s->go.end()) {
		    nsid = pos->second;
		    states[nsid]->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 10); // level 10
		    states.push_back(ns);
		    s->go.insert(pair<int, int>(v, ns->id));
		    ns->outputs.insert(ptn->idx);
		}
	    }

	    ++oite;
	}

	map<int, int>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = states[goite->second];
	    if (ignore_ptns.size() > 0) {
		ns->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    ++goite;
	    next_layer->insert(ns);
	    if (s == ipa_0_out_fs+8)
		break;
	}

	if (s != ipa_0_out_fs+8)
	{
	    State *fs;

	    if (ignore_ptns.size() == 0)
		fs = ipa_0_out_fs+9;
	    else {
		fs = new State(sid++, 9+1);
		states.push_back(fs);
		fs->fail = true;
		fs->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    for (int v=0; v<= PORT_MASK; v++) {
		if (s->go.find(v) == s->go.end())
		    s->go.insert(pair<int, int>(v, fs->id));
	    }

	    next_layer->insert(fs);
	}
	
	++site;
    }

    tmplp = layer;
    layer = next_layer;
    next_layer = tmplp;
    next_layer->clear();

    // Proto:
    site = layer->begin();
    while (site != layer->end())
    {
	s = *site;
	int nsid = 0;
	set<int> ignore_ptns;

	set<int>::iterator oite = s->outputs.begin();
	while (oite != s->outputs.end())
	{
	    g4c_pattern_t *ptn = ptns+(*oite);
	    if (ptn->proto < 0 || ptn->proto > PROTO_MASK) {
		ignore_ptns.insert(ptn->idx);
	    } else {
		int v = get_eproto(ptn->proto);
		map<int, int>::iterator pos = s->go.find(v);
		if (pos != s->go.end()) {
		    nsid = pos->second;
		    states[nsid]->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 11); // level 11
		    states.push_back(ns);
		    s->go.insert(pair<int, int>(v, ns->id));
		    ns->outputs.insert(ptn->idx);
		}
	    }

	    ++oite;
	}

	map<int, int>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = states[goite->second];
	    if (ignore_ptns.size() > 0) {
		ns->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    ++goite;
	    next_layer->insert(ns);
	    if (s == ipa_0_out_fs+9)
		break;
	}

	if (s != ipa_0_out_fs+9)
	{
	    State *fs;

	    if (ignore_ptns.size() == 0)
		fs = ipa_0_out_fs+10;
	    else {
		fs = new State(sid++, 10+1);
		states.push_back(fs);
		fs->fail = true;
		fs->outputs.insert(ignore_ptns.begin(), ignore_ptns.end());
	    }

	    for (int v=0; v<= PROTO_MASK; v++) {
		if (s->go.find(v) == s->go.end())
		    s->go.insert(pair<int, int>(v, fs->id));
	    }

	    next_layer->insert(fs);
	}
	
	++site;
    }

    layer->clear();
    delete layer;

    cl->last_layer = next_layer;
    return sid;
}

static void
_dump_state(State *s)
{
    assert(s);
    printf("State id %d, fail %d, failon %d, level %d\n",
	   s->id, s->fail, s->failon, s->level);

    set<int>::iterator oite = s->outputs.begin();
    printf("  Outputs: ");
    while (oite != s->outputs.end()) {
	printf("%d ", *oite);
	++oite;
    }
    printf("\n");
    
    map<int, int>::iterator goite = s->go.begin();
    int i = 0;
    printf("  Gotos: ");
    while (goite != s->go.end()) {
	if (i && ((i&0x7) == 0)) {
	    printf("         0x%04X => %-6d ", goite->first, goite->second);
	} else
	    printf("0x%04X => %-6d ", goite->first, goite->second);

	if ((i&0x7) == 7)
	    printf("\n");
	i++;
	goite++;
    }
    
    if ((i&0x7) != 0)
	printf("\n");
}

static State *
_match_pattern(g4c_pattern_t *ent, Classifier *cl)
{
    State *s = cl->states[0];
    
    for (int b=0; b<4; b++) {
	int v = (ent->src_addr>>(b*8))&0xff;
	printf("Find 0x%X in src addr %d\n", v, b);
	map<int, int>::iterator ite = s->go.find(v);
	_dump_state(s);
	if (ite != s->go.end()) {
	    int nsid = ite->second;
	    s = cl->states[nsid];
	} else {
	    fprintf(stderr, "No goto at state %d\n", s->id);
	    return 0;
	}
    }

    for (int b=0; b<4; b++) {
	int v = (ent->dst_addr>>(b*8))&0xff;
	printf("Find 0x%X in dst addr %d\n", v, b);
	map<int, int>::iterator ite = s->go.find(v);
	_dump_state(s);
	if (ite != s->go.end()) {
	    int nsid = ite->second;
	    s = cl->states[nsid];
	} else {
	    fprintf(stderr, "No goto at state %d\n", s->id);
	    return 0;
	}
    }
    
    do {
	int pv;
	printf("Find 0x%X in src port\n", ent->src_port);
	if (get_eport(ent->src_port) != ent->src_port) {
	    pv = 0;
	} else
	    pv = get_eport(ent->src_port);

	map<int, int>::iterator ite = s->go.find(pv);
	_dump_state(s);
	if (ite != s->go.end()) {
	    int nsid = ite->second;
	    s = cl->states[nsid];
	} else {
	    fprintf(stderr, "No goto at state %d\n", s->id);
	    return 0;
	}
    } while (0);

    do {
	int pv;
	printf("Find 0x%X in dst port\n", ent->dst_port);
	if (get_eport(ent->dst_port) != ent->dst_port) {
	    pv = 0;
	} else
	    pv = get_eport(ent->dst_port);

	map<int, int>::iterator ite = s->go.find(pv);
	_dump_state(s);
	if (ite != s->go.end()) {
	    int nsid = ite->second;
	    s = cl->states[nsid];
	} else {
	    fprintf(stderr, "No goto at state %d\n", s->id);
	    return 0;
	}
    } while (0);

    do {
	int pv;
	printf("Find 0x%X in protocol\n", ent->proto);

	if (get_eproto(ent->proto) != ent->proto) {
	    pv = 0;
	} else
	    pv = get_eproto(ent->proto);

	map<int, int>::iterator ite = s->go.find(pv);
	_dump_state(s);
	if (ite != s->go.end()) {
	    int nsid = ite->second;
	    s = cl->states[nsid];
	} else {
	    fprintf(stderr, "No goto at state %d\n", s->id);
	    return 0;
	}
    } while (0);

    _dump_state(s);

    return s;   
}

#include <time.h>
static void
_gen_rand_ptns(g4c_pattern_t *ptns, int n)
{
    srand(clock());

    for (int i=0; i<n; i++) {
	int nbits = rand()%5;
	ptns[i].nr_src_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].src_addr = (ptns[i].src_addr<<8)|(rand()&0xff);
	nbits = rand()%5;
	ptns[i].nr_dst_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].dst_addr = (ptns[i].dst_addr<<8)|(rand()&0xff);
	ptns[i].src_port = rand()%(2*(PORT_MASK)) - (PORT_MASK+1);
	ptns[i].dst_port = rand()%(2*(PORT_MASK)) - (PORT_MASK+1);
	ptns[i].proto = rand()%(2*PROTO_MASK) - (PROTO_MASK+1);
	ptns[i].idx = i;
    }
}

int main(int argc, char *argv[])
{
    int np = 100;

    if (argc > 1)
	np = atoi(argv[1]);
    
    g4c_pattern_t ptns[10];
    for (int i=0; i<10; i++)
	ptns[i].idx = i;

    g4c_pattern_t *pptns = new g4c_pattern_t[np];
    assert(pptns);
    bzero(pptns, sizeof(g4c_pattern_t)*np);

    _gen_rand_ptns(pptns, np);

    Classifier cl;

    ptns[0].src_addr = 0x00151617;
    ptns[0].nr_src_netbits = 16;
    ptns[0].dst_addr = 0x11121314;
    ptns[0].nr_dst_netbits = 32;
    ptns[0].src_port = 0x3f;
    ptns[0].dst_port = 0x99;
    ptns[0].proto = 0x10;

    ptns[1].src_addr = 0x00151617;
    ptns[1].nr_src_netbits = 8;
    ptns[1].dst_addr = 0x11121314;
    ptns[1].nr_dst_netbits = 0;
    ptns[1].src_port = 0x803f;
    ptns[1].dst_port = 0x02;
    ptns[1].proto = 0x8010;
    
    ptns[2].src_addr = 0x00151917;
    ptns[2].nr_src_netbits = 8;
    ptns[2].dst_addr = 0x11121314;
    ptns[2].nr_dst_netbits = 24;
    ptns[2].src_port = 0x3f;
    ptns[2].dst_port = 0x199;
    ptns[2].proto = 0x4510;

    ptns[3].src_addr = 0x00151617;
    ptns[3].nr_src_netbits = 16;
    ptns[3].dst_addr = 0x11121314;
    ptns[3].nr_dst_netbits = 32;
    ptns[3].src_port = 0x3f;
    ptns[3].dst_port = 0x49;
    ptns[3].proto = 0x110;

    ptns[4].src_addr = 0x00151617;
    ptns[4].nr_src_netbits = 24;
    ptns[4].dst_addr = 0x11121314;
    ptns[4].nr_dst_netbits = 8;
    ptns[4].src_port = 0x3f;
    ptns[4].dst_port = 0x299;
    ptns[4].proto = 0x8110;

    ptns[5].src_addr = 0x00151617;
    ptns[5].nr_src_netbits = 0;
    ptns[5].dst_addr = 0x11121314;
    ptns[5].nr_dst_netbits = 32;
    ptns[5].src_port = 0x3f;
    ptns[5].dst_port = 0x99;
    ptns[5].proto = 0x10;

    ptns[6].src_addr = 0x00151617;
    ptns[6].nr_src_netbits = 0;
    ptns[6].dst_addr = 0x11121314;
    ptns[6].nr_dst_netbits = 0;
    ptns[6].src_port = 0x3f;
    ptns[6].dst_port = 0x99;
    ptns[6].proto = 0x10;

    ptns[7].src_addr = 0x00151617;
    ptns[7].nr_src_netbits = 0;
    ptns[7].dst_addr = 0x11121314;
    ptns[7].nr_dst_netbits = 0;
    ptns[7].src_port = 0x893f;
    ptns[7].dst_port = 0x99;
    ptns[7].proto = 0x10;

    ptns[8].src_addr = 0x00151617;
    ptns[8].nr_src_netbits = 16;
    ptns[8].dst_addr = 0x11121314;
    ptns[8].nr_dst_netbits = 32;
    ptns[8].src_port = 0x3f;
    ptns[8].dst_port = 0x8199;
    ptns[8].proto = 0x8110;

    int ns = _build_goto(pptns, np, &cl);
    printf(" %d states in total\n", ns);

    State *rt;
    /*rt = _match_pattern(ptns+6, &cl);
    if (rt->outputs.size() > 0)
	printf("\nFound\n");
    else
	printf("\nFailed\n");*/

    rt = _match_pattern(pptns+(np/2), &cl);
    if (rt->outputs.size() > 0)
	printf("\nFound %d\n", np/2);
    else
	printf("\nFailed %d\n", np/2);
    
    
    return 0;
}
