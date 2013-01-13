#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <map>
#include <set>

#include "g4c.hh"
#include "g4c_cl.h"

using namespace std;

static int verbose_level = 1;

extern "C" void
g4c_cl_init(int n, char *keys, int *values)
{
    for (int i=0; i<n; i++) {
	switch(keys[i]) {
	case G4C_CL_VERBOSE:
	    verbose_level = values[i];
	    break;
	default:
	    break;
	}
    }
}

typedef int ptn_idx_t;
typedef int goto_key_t;

class State {
public:
    int id;
    bool fail;
    bool failon;
    map<goto_key_t, State*> go;
    set<ptn_idx_t> outputs;
    bool final;
    int level;
    State(int sid=0, int slevel=0) : id(sid), fail(false), final(false), failon(false) {
	level = slevel;}    
};

class Classifier {
public:
    Classifier() {
	nr_rules = 0;
	sa_ly = 0;
	da_ly = 0;
	sp_ly = 0;
	dp_ly = 0;
	p_ly = 0;	
    }

    ~Classifier() {
	if (sa_ly)
	    delete sa_ly;
	if (da_ly)
	    delete da_ly;
	if (sp_ly)
	    delete sp_ly;
	if (dp_ly)
	    delete dp_ly;
	if (p_ly)
	    delete p_ly;
	sass.clear();
	dass.clear();
	spss.clear();
	dpss.clear();
	pss.clear();
	for (int i=0; i<states.size(); i++)
	    delete states[i];
	states.clear();
    }
    int nr_rules;
    vector<State*> states;
    vector<State*> sass;
    set<State*> *sa_ly;
    vector<State*> dass;
    set<State*> *da_ly;
    vector<State*> spss;
    set<State*> *sp_ly;
    vector<State*> dpss;
    set<State*> *dp_ly;
    vector<State*> pss;
    set<State*> *p_ly;
};


inline static void
_bmp_set_bit(uint32_t *bmp, int which)
{
    div_t q = div(which, 32);
    bmp[q.quot] |= (0x1<<q.rem);
}

inline static void
_fill_res_bitmap(uint32_t *bmp, set<ptn_idx_t>* sres)
{
    set<ptn_idx_t>::iterator ite = sres->begin();
    while (ite != sres->end()) {
	_bmp_set_bit(bmp, *ite);
	++ite;
    }
}

inline static void
_fill_states(
    vector<State*> &ss,
    int *trans_bp,
    uint32_t state_size,
    uint32_t *ress_bp,
    uint32_t rstride)
{
    for (int i=0; i<ss.size(); i++) {
	State* s = ss[i];
	assert(s->go.size() == 0 || s->go.size() == state_size);
	
	int *trans = cl_trans_tbl(trans_bp, s->id, state_size);
	for (int v=0; v<state_size; v++)
	    trans[v] = s->go[v]->id;

	_fill_res_bitmap(cl_res(ress_bp, s->id, rstride), &s->outputs);
    }
}

static g4c_classifier_t*
_g4c_cvt_cl(Classifier *cl, int with_dev, int stream)
{
    size_t res_sz = (g4c_round_up(cl->nr_rules, G4C_CL_RES_SZ_ALIGN*8))/8;
    size_t sa_sz =
	g4c_round_up(cl->sass.size()*G4C_IPA_STATE_SIZE*sizeof(g4c_cl_sid_t), G4C_PAGE_SIZE) +
	g4c_round_up(cl->sass.size()*res_sz, G4C_PAGE_SIZE);
    size_t da_sz =
	g4c_round_up(cl->dass.size()*G4C_IPA_STATE_SIZE*sizeof(g4c_cl_sid_t), G4C_PAGE_SIZE) +
	g4c_round_up(cl->dass.size()*res_sz, G4C_PAGE_SIZE);
    size_t sp_sz =
	g4c_round_up(cl->spss.size()*PORT_STATE_SIZE*sizeof(g4c_cl_sid_t), G4C_PAGE_SIZE) +
	g4c_round_up(cl->spss.size()*res_sz, G4C_PAGE_SIZE);
    size_t dp_sz =
	g4c_round_up(cl->dpss.size()*PORT_STATE_SIZE*sizeof(g4c_cl_sid_t), G4C_PAGE_SIZE) +
	g4c_round_up(cl->dpss.size()*res_sz, G4C_PAGE_SIZE);
    size_t pt_sz =
	g4c_round_up(cl->pss.size()*PROTO_STATE_SIZE*sizeof(g4c_cl_sid_t), G4C_PAGE_SIZE) +
	g4c_round_up(cl->pss.size()*res_sz, G4C_PAGE_SIZE);

    size_t msz = sa_sz + da_sz + sp_sz + dp_sz + pt_sz + G4C_PAGE_SIZE;
    g4c_classifier_t *gcl = (g4c_classifier_t*)g4c_alloc_page_lock_mem(msz);

    g4c_classifier_t *dgcl = 0;
    if (with_dev) {
	dgcl = (g4c_classifier_t*)g4c_alloc_dev_mem(msz);
    }
    if (!gcl || (with_dev && !dgcl))
	return 0;

    memset(gcl, 0, msz);

    gcl->mem = (void*)gcl;
    gcl->memsz = msz;
    gcl->nrules = cl->nr_rules;
    gcl->res_sz = res_sz;
    gcl->res_stride = res_sz/sizeof(uint32_t);

    gcl->nr_saddr_sts = cl->sass.size();
    gcl->nr_daddr_sts = cl->dass.size();
    gcl->nr_sp_sts = cl->spss.size();
    gcl->nr_dp_sts = cl->dpss.size();
    gcl->nr_pt_sts = cl->pss.size();

    gcl->saddr_trs = (int*)g4c_ptr_add(gcl->mem, G4C_PAGE_SIZE);
    gcl->saddr_ress = (uint32_t*)g4c_ptr_add(gcl->saddr_trs,
				  g4c_round_up(cl->sass.size()*G4C_IPA_STATE_SIZE*sizeof(g4c_cl_sid_t),
					       G4C_PAGE_SIZE));

    gcl->daddr_trs = (int*)g4c_ptr_add(gcl->saddr_trs, sa_sz);
    gcl->daddr_ress = (uint32_t*)g4c_ptr_add(gcl->daddr_trs,
				  g4c_round_up(cl->dass.size()*G4C_IPA_STATE_SIZE*sizeof(g4c_cl_sid_t),
					       G4C_PAGE_SIZE));

    gcl->sp_trs = (int*)g4c_ptr_add(gcl->daddr_trs, da_sz);
    gcl->sp_ress = (uint32_t*)g4c_ptr_add(gcl->sp_trs,
			       g4c_round_up(cl->spss.size()*PORT_STATE_SIZE*sizeof(g4c_cl_sid_t),
					    G4C_PAGE_SIZE));

    gcl->dp_trs = (int*)g4c_ptr_add(gcl->sp_trs, sp_sz);
    gcl->dp_ress = (uint32_t*)g4c_ptr_add(gcl->dp_trs,
			       g4c_round_up(cl->dpss.size()*PORT_STATE_SIZE*sizeof(g4c_cl_sid_t),
					    G4C_PAGE_SIZE));

    gcl->pt_trs = (int*)g4c_ptr_add(gcl->dp_trs, dp_sz);
    gcl->pt_ress = (uint32_t*)g4c_ptr_add(gcl->pt_trs,
			       g4c_round_up(cl->pss.size()*PROTO_STATE_SIZE*sizeof(g4c_cl_sid_t),
					    G4C_PAGE_SIZE));
    
    _fill_states(cl->sass, gcl->saddr_trs, G4C_IPA_STATE_SIZE, gcl->saddr_ress, gcl->res_stride);
    _fill_states(cl->dass, gcl->daddr_trs, G4C_IPA_STATE_SIZE, gcl->daddr_ress, gcl->res_stride);
    
    _fill_states(cl->spss, gcl->sp_trs, PORT_STATE_SIZE, gcl->sp_ress, gcl->res_stride);
    _fill_states(cl->dpss, gcl->dp_trs, PORT_STATE_SIZE, gcl->dp_ress, gcl->res_stride);

    _fill_states(cl->pss, gcl->pt_trs, PROTO_STATE_SIZE, gcl->pt_ress, gcl->res_stride);

    if (with_dev) {
	gcl->devmem = dgcl;
	gcl->dev_saddr_trs = (int*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->saddr_trs, gcl->mem));
	gcl->dev_daddr_trs = (int*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->daddr_trs, gcl->mem));
	gcl->dev_sp_trs = (int*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->sp_trs, gcl->mem));
	gcl->dev_dp_trs = (int*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->dp_trs, gcl->mem));
	gcl->dev_pt_trs = (int*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->pt_trs, gcl->mem));

	gcl->dev_saddr_ress = (uint32_t*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->saddr_ress, gcl->mem));
	gcl->dev_daddr_ress = (uint32_t*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->daddr_ress, gcl->mem));
	gcl->dev_sp_ress = (uint32_t*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->sp_ress, gcl->mem));
	gcl->dev_dp_ress = (uint32_t*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->dp_ress, gcl->mem));
	gcl->dev_pt_ress = (uint32_t*)g4c_ptr_add(gcl->devmem, g4c_ptr_offset(gcl->pt_ress, gcl->mem));

	g4c_h2d_async(gcl->mem, gcl->devmem, gcl->memsz, stream);
	g4c_stream_sync(stream);
    }
    return gcl;
}


static int
_g4c_cl_build(g4c_pattern_t *ptns, int n, Classifier *cl)
{
    int sid = 0;

    vector<State*> &states = cl->states;

    cl->nr_rules = n;
    states.reserve(n<<2);

    do { // Build Src addr trie
	State *s = new State(sid++, 0);
	vector<State*> &sass = cl->sass;

	states.push_back(s);
	sass.push_back(s);

	for (int i=0; i<n; i++) {
	    g4c_pattern_t *ptn = ptns+i;

	    s = sass[0];
	    for (int b=0; b<4; b++) {
		if ((ptn->nr_src_netbits>>3) > b) // normal
		{
		    goto_key_t v = (ptn->src_addr>>(b*8)) & 0xff;
		    map<goto_key_t, State*>::iterator ite = s->go.find(v);
		    if (ite != s->go.end()) { // go through existing path
			s = ite->second;
		    } else { // need a new path
			State *ns = new State(sid++, b+1);
			states.push_back(ns);
			sass.push_back(ns);
			s->go.insert(pair<goto_key_t, State*>(v, ns));
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

	State *ipa_0_out_fs = new State[4]; // shared fail nodes with 0 outputs
	for (int l=0; l<4; l++) {
	    ipa_0_out_fs[l].id = sid++;
	    ipa_0_out_fs[l].fail = true;
	    ipa_0_out_fs[l].level = l+1;
	    states.push_back(ipa_0_out_fs+l);
	    sass.push_back(ipa_0_out_fs+l);

	    if (l != 3)
		for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		    ipa_0_out_fs[l].go.insert(
			pair<goto_key_t, State*>(v, ipa_0_out_fs+l+1));
		}
	}

	s = sass[0];
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
		    map<goto_key_t, State*>::iterator goite = curs->go.begin();
		    while(goite != curs->go.end()) {
			State *ns = goite->second;
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
		    } else if (!curs->fail) {			
			fs = new State(sid++, l+1);
			states.push_back(fs);
			sass.push_back(fs);
			fs->fail = true;
			fs->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		    } else
			fs = curs;
		
		    for (goto_key_t v=0; v<G4C_IPA_STATE_SIZE; v++) {
			if (curs->go.find(v) == curs->go.end()) // failure value
			{
			    curs->go.insert(pair<goto_key_t, State*>(v, fs));
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
		   layer->size(), sass.size());
	}

	cl->sa_ly = layer;
	delete next_layer;    
    } while (0);

    do { // Build Dst addr trie
	sid = 0;
	State *s = new State(sid++, 0);
	vector<State*> &dass = cl->dass;

	states.push_back(s);
	dass.push_back(s);

	for (int i=0; i<n; i++) {
	    g4c_pattern_t *ptn = ptns+i;

	    s = dass[0];
	    for (int b=0; b<4; b++) {
		if ((ptn->nr_dst_netbits>>3) > b) // normal
		{
		    goto_key_t v = (ptn->dst_addr>>(b*8)) & 0xff;
		    map<goto_key_t, State*>::iterator ite = s->go.find(v);
		    if (ite != s->go.end()) { // go through existing path
			s = ite->second;
		    } else { // need a new path
			State *ns = new State(sid++, b+1);
			states.push_back(ns);
			dass.push_back(ns);
			s->go.insert(pair<goto_key_t, State*>(v, ns));
			s = ns;				 
		    }
		}
		else if ((ptn->nr_dst_netbits>>3) == b) // fail now
		{
		    s->outputs.insert(ptn->idx);
		    s->failon = true;
		    break;
		} else {
		    fprintf(stderr, "dst ip bug: b: %d, nr_dst_netbits: %d\n",
			    b, ptn->nr_dst_netbits);
		    assert(0);
		}
	    }
	    if (ptn->nr_dst_netbits == 32)
		s->outputs.insert(ptn->idx);
	}

	State *ipa_0_out_fs = new State[4]; // shared fail nodes with 0 outputs
	for (int l=0; l<4; l++) {
	    ipa_0_out_fs[l].id = sid++;
	    ipa_0_out_fs[l].fail = true;
	    ipa_0_out_fs[l].level = l+1;
	    states.push_back(ipa_0_out_fs+l);
	    dass.push_back(ipa_0_out_fs+l);

	    if (l != 3)
		for (int v=0; v<G4C_IPA_STATE_SIZE; v++) {
		    ipa_0_out_fs[l].go.insert(
			pair<goto_key_t, State*>(v, ipa_0_out_fs+l+1));
		}
	}

	s = dass[0];
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
		    map<goto_key_t, State*>::iterator goite = curs->go.begin();
		    while(goite != curs->go.end()) {
			State *ns = goite->second;
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
		    } else if (!curs->fail) {
			fs = new State(sid++, l+1);
			states.push_back(fs);
			dass.push_back(fs);
			fs->fail = true;
			fs->outputs.insert(curs->outputs.begin(), curs->outputs.end());
		    } else
			fs = curs;
		
		    for (goto_key_t v=0; v<G4C_IPA_STATE_SIZE; v++) {
			if (curs->go.find(v) == curs->go.end()) // failure value
			{
			    curs->go.insert(pair<goto_key_t, State*>(v, fs));
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
	    printf("INFO:: G4C Classifier: build_goto: has %lu dst finals, %lu dst nodes\n",
		   layer->size(), dass.size());
	}

	cl->da_ly = layer;
	delete next_layer;    
    } while (0);

    do { // Build for src port
	sid = 0;
	State *s = new State(sid++, 0);
	vector<State*> &spss = cl->spss;

	states.push_back(s);
	spss.push_back(s);	

	for (int i=0; i<n; i++) {
	    g4c_pattern_t *ptn = ptns+i;

	    if (ptn->src_port < 0 || ptn->src_port > PORT_MASK) {
		s->outputs.insert(ptn->idx);
	    } else {
		goto_key_t v = get_eport(ptn->src_port);
		map<goto_key_t, State*>::iterator ite = s->go.find(v);
		if (ite != s->go.end()) {
		    ite->second->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 0);
		    states.push_back(ns);
		    spss.push_back(ns);
		    s->go.insert(pair<goto_key_t, State*>(v, ns));
		    ns->outputs.insert(ptn->idx);
		}				 
	    }
	}

	set<State*> *next_layer = new set<State*>();
	assert(next_layer);

	map<goto_key_t, State*>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = goite->second;
	    ns->outputs.insert(s->outputs.begin(), s->outputs.end());
	    ++goite;
	    next_layer->insert(ns);
	}

	if (s->go.size() <= PORT_MASK) {
	    State *fs = new State(sid++, 1);
	    states.push_back(fs);
	    spss.push_back(fs);
	    fs->fail = true;
	    fs->outputs.insert(s->outputs.begin(), s->outputs.end());

	    for (goto_key_t v=0; v<=PORT_MASK; v++)
		if (s->go.find(v) == s->go.end())
		    s->go.insert(make_pair(v, fs));
	    next_layer->insert(fs);
	}

	if (verbose_level > 0) {
	    printf("INFO:: G4C Classifier: build_goto: has %lu src port finals, %lu src port nodes\n",
		   next_layer->size(), spss.size());
	}

	cl->sp_ly = next_layer;	
    } while (0);


    
    do { // Build for dst port
	sid = 0;
	State *s = new State(sid++, 0);
	vector<State*> &dpss = cl->dpss;

	states.push_back(s);
	dpss.push_back(s);	

	for (int i=0; i<n; i++) {
	    g4c_pattern_t *ptn = ptns+i;

	    if (ptn->dst_port < 0 || ptn->dst_port > PORT_MASK) {
		s->outputs.insert(ptn->idx);
	    } else {
		goto_key_t v = get_eport(ptn->dst_port);
		map<goto_key_t, State*>::iterator ite = s->go.find(v);
		if (ite != s->go.end()) {
		    ite->second->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 0);
		    states.push_back(ns);
		    dpss.push_back(ns);
		    s->go.insert(pair<goto_key_t, State*>(v, ns));
		    ns->outputs.insert(ptn->idx);
		}				 
	    }
	}

	set<State*> *next_layer = new set<State*>();
	assert(next_layer);

	map<goto_key_t, State*>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = goite->second;
	    ns->outputs.insert(s->outputs.begin(), s->outputs.end());
	    ++goite;
	    next_layer->insert(ns);
	}

	if (s->go.size() <= PORT_MASK) {
	    State *fs = new State(sid++, 1);
	    states.push_back(fs);
	    dpss.push_back(fs);
	    fs->fail = true;
	    fs->outputs.insert(s->outputs.begin(), s->outputs.end());

	    for (goto_key_t v=0; v<=PORT_MASK; v++)
		if (s->go.find(v) == s->go.end())
		    s->go.insert(make_pair(v, fs));
	    next_layer->insert(fs);
	}

	if (verbose_level > 0) {
	    printf("INFO:: G4C Classifier: build_goto: has %lu dst port finals, %lu dst port nodes\n",
		   next_layer->size(), dpss.size());
	}

	cl->dp_ly = next_layer;	
    } while (0);


    
    do { // Build for proto
	sid = 0;
	State *s = new State(sid++, 0);
	vector<State*> &pss = cl->pss;

	states.push_back(s);
	pss.push_back(s);	

	for (int i=0; i<n; i++) {
	    g4c_pattern_t *ptn = ptns+i;

	    if (ptn->proto < 0 || ptn->proto > PROTO_MASK) {
		s->outputs.insert(ptn->idx);
	    } else {
		goto_key_t v = get_eproto(ptn->proto);
		map<goto_key_t, State*>::iterator ite = s->go.find(v);
		if (ite != s->go.end()) {
		    ite->second->outputs.insert(ptn->idx);
		} else {
		    State *ns = new State(sid++, 0);
		    states.push_back(ns);
		    pss.push_back(ns);
		    s->go.insert(pair<goto_key_t, State*>(v, ns));
		    ns->outputs.insert(ptn->idx);
		}				 
	    }
	}

	set<State*> *next_layer = new set<State*>();
	assert(next_layer);

	map<goto_key_t, State*>::iterator goite = s->go.begin();
	while (goite != s->go.end()) {
	    State *ns = goite->second;
	    ns->outputs.insert(s->outputs.begin(), s->outputs.end());
	    ++goite;
	    next_layer->insert(ns);
	}

	if (s->go.size() <= PROTO_MASK) {
	    State *fs = new State(sid++, 1);
	    states.push_back(fs);
	    pss.push_back(fs);
	    fs->fail = true;
	    fs->outputs.insert(s->outputs.begin(), s->outputs.end());

	    for (goto_key_t v=0; v<=PROTO_MASK; v++)
		if (s->go.find(v) == s->go.end())
		    s->go.insert(make_pair(v, fs));
	    next_layer->insert(fs);
	}

	if (verbose_level > 0) {
	    printf("INFO:: G4C Classifier: build_goto: has %lu proto finals, %lu proto nodes\n",
		   next_layer->size(), pss.size());
	}

	cl->p_ly = next_layer;	
    } while (0);

    return states.size();
}

extern "C" g4c_classifier_t *
g4c_create_classifier(g4c_pattern_t *ptn, int nptn, int create_dev, int stream)
{
    Classifier cl;
    _g4c_cl_build(ptn, nptn, &cl);

    g4c_classifier_t *gcl = _g4c_cvt_cl(&cl, create_dev, stream);
    return gcl;
}

extern "C" int
g4c_cpu_classify_pkt(g4c_classifier_t *gcl, uint8_t *ttlptr)
{
    uint32_t *r[5];
    uint8_t pt = ttlptr[0] & PROTO_MASK;
    int rid = gcl->pt_trs[pt];
    r[0] = cl_res(gcl->pt_ress, rid, gcl->res_stride);

    
    uint32_t ipa = *(uint32_t*)(ttlptr+3);    
    rid = *cl_ipa_trans(gcl->saddr_trs, 0, (ipa)&0xff);
    rid = *cl_ipa_trans(gcl->saddr_trs, rid, (ipa>>8)&0xff);
    rid = *cl_ipa_trans(gcl->saddr_trs, rid, (ipa>>16)&0xff);
    rid = *cl_ipa_trans(gcl->saddr_trs, rid, (ipa>>24)&0xff);	
    r[1] = cl_res(gcl->saddr_ress, rid, gcl->res_stride);

    
    ipa = *(uint32_t*)(ttlptr+7);    
    rid = *cl_ipa_trans(gcl->daddr_trs, 0, (ipa)&0xff);
    rid = *cl_ipa_trans(gcl->daddr_trs, rid, (ipa>>8)&0xff);
    rid = *cl_ipa_trans(gcl->daddr_trs, rid, (ipa>>16)&0xff);
    rid = *cl_ipa_trans(gcl->daddr_trs, rid, (ipa>>24)&0xff);	
    r[2] = cl_res(gcl->daddr_ress, rid, gcl->res_stride);

    ipa = (*(uint32_t*)(ttlptr+11));
	
    rid = gcl->sp_trs[ipa & 0xffff];
    r[3] = cl_res(gcl->sp_ress, rid, gcl->res_stride);
    rid = gcl->dp_trs[ipa>>16];
    r[4] = cl_res(gcl->dp_ress, rid, gcl->res_stride);

    for (int i=0; i<gcl->res_stride; i++) {
	uint32_t v = r[0][i] & r[1][i];
	if (v) {
	    v &= r[2][i] & r[3][i] & r[4][i];
	    if (v) {
		return __builtin_ffs(v)-1 + i*32;
	    }
	}
    }
    
    return -1;
}

#define CL_PKTS_PER_BLK 32
#define CL_BLK_SIZE (CL_PKTS_PER_BLK*4)

__global__ void
gpu_cl_0(g4c_classifier_t *gcl, uint8_t *data, uint32_t stride, uint32_t ttl_ofs,
	 int *ress, uint32_t res_stride, uint32_t res_ofs)
{
    __shared__ uint32_t* comp_ress[CL_BLK_SIZE];
    int pktid = threadIdx.x + blockIdx.x*blockDim.x;
    uint8_t *pkt = data + stride*pktid;

    if (threadIdx.y == 0)
    {
	uint8_t pt = pkt[ttl_ofs + 1] & PROTO_MASK;
	int rid = gcl->dev_pt_trs[pt];
	comp_ress[threadIdx.x] = cl_res(gcl->dev_pt_ress, rid, gcl->res_stride);	
    }
    else if (threadIdx.y == 1)
    {
	uint32_t sa = *(uint32_t*)(pkt+ttl_ofs+4);
	int nid = 0;

	nid = *cl_ipa_trans(gcl->dev_saddr_trs, 0, (sa)&0xff);
	nid = *cl_ipa_trans(gcl->dev_saddr_trs, nid, (sa>>8)&0xff);
	nid = *cl_ipa_trans(gcl->dev_saddr_trs, nid, (sa>>16)&0xff);
	nid = *cl_ipa_trans(gcl->dev_saddr_trs, nid, (sa>>24)&0xff);
	
	comp_ress[threadIdx.x + CL_PKTS_PER_BLK] = cl_res(gcl->dev_saddr_ress, nid, gcl->res_stride);	
    }
    else if (threadIdx.y == 2)
    {
	uint32_t da = *(uint32_t*)(pkt+ttl_ofs+8);
	int nid = 0;
	
	nid = *cl_ipa_trans(gcl->dev_daddr_trs, 0, (da)&0xff);
	nid = *cl_ipa_trans(gcl->dev_daddr_trs, nid, (da>>8)&0xff);
	nid = *cl_ipa_trans(gcl->dev_daddr_trs, nid, (da>>16)&0xff);
	nid = *cl_ipa_trans(gcl->dev_daddr_trs, nid, (da>>24)&0xff);
	
	comp_ress[threadIdx.x + (CL_PKTS_PER_BLK<<1)] = cl_res(gcl->dev_daddr_ress, nid, gcl->res_stride);
    }
    else if (threadIdx.y == 3)
    {
	uint32_t p = (*(uint32_t*)(pkt+ttl_ofs+12));
	
	int rid = gcl->dev_sp_trs[p & 0xffff];
	comp_ress[threadIdx.x + (CL_PKTS_PER_BLK*3)] = cl_res(gcl->dev_sp_ress, rid, gcl->res_stride);
	rid = gcl->dev_dp_trs[p>>16];
	comp_ress[threadIdx.x + (CL_PKTS_PER_BLK<<2)] = cl_res(gcl->dev_dp_ress, rid, gcl->res_stride);
    }

    __syncthreads();

    if (threadIdx.y==0) {
	uint32_t *r[5];
	r[0] = comp_ress[threadIdx.x];
	r[1] = comp_ress[threadIdx.x + CL_PKTS_PER_BLK];
	r[2] = comp_ress[threadIdx.x + CL_PKTS_PER_BLK<<1];
	r[3] = comp_ress[threadIdx.x + CL_PKTS_PER_BLK*3];
	r[4] = comp_ress[threadIdx.x + CL_PKTS_PER_BLK<<2];

	#pragma unroll
	for (int i=0; i<gcl->res_stride; i++) {
	    uint32_t v = r[0][i] & r[1][i] & r[2][i] & r[3][i] & r[4][i];;
	    if (v) {
		*(ress + pktid*res_stride + res_ofs) = __ffs(v)-1 + i*32;
		return;
	    }		
	}
	*(ress + pktid*res_stride + res_ofs) = -1;
    }
    
}

extern "C" int
g4c_gpu_classify_pkts(g4c_classifier_t *dgcl, int npkts,
		      uint8_t *data, uint32_t stride, uint32_t ttl_ofs, uint32_t l3hdr_ofs,
		      int *ress, uint32_t res_stride, uint32_t res_ofs,
		      int stream)
{
    dim3 blockdim(CL_PKTS_PER_BLK, 4);
    int ng = g4c_round_up(npkts, CL_PKTS_PER_BLK);
    ng /= CL_PKTS_PER_BLK;
    dim3 griddim(ng, 1);
    cudaStream_t s = g4c_get_stream(stream);

    gpu_cl_0<<<griddim, blockdim, 0, s>>>(dgcl, data, stride, ttl_ofs, ress, res_stride, res_ofs);
    return 0;
}


static void
_dump_state(State *s)
{
    assert(s);
    printf("State id %d, fail %d, failon %d, level %d\n",
	   s->id, s->fail, s->failon, s->level);

    set<ptn_idx_t>::iterator oite = s->outputs.begin();
    printf("  Outputs: ");
    while (oite != s->outputs.end()) {
	printf("%d ", *oite);
	++oite;
    }
    printf("\n");
    
    map<goto_key_t, State*>::iterator goite = s->go.begin();
    int i = 0;
    printf("  Gotos: ");
    while (goite != s->go.end()) {
	if (i && ((i&0x7) == 0)) {
	    printf("         0x%04X => %-6d ", goite->first, goite->second->id);
	} else
	    printf("0x%04X => %-6d ", goite->first, goite->second->id);

	if ((i&0x7) == 7)
	    printf("\n");
	i++;
	goite++;
    }
    
    if ((i&0x7) != 0)
	printf("\n");
}

static set<ptn_idx_t> *
_match_pattern(g4c_pattern_t *ent, Classifier *cl)
{
    State *sa, *da, *sp, *dp, *proto;
    
    do {
	sa = cl->sass[0];
	for (int b=0; b<4; b++) {
	    int v = (ent->src_addr>>(b*8))&0xff;
	    printf("Find 0x%X in src addr %d\n", v, b);
	    map<goto_key_t, State*>::iterator ite = sa->go.find(v);
	    if (verbose_level > 2)
		_dump_state(sa);
	    if (ite != sa->go.end()) {
		sa = ite->second;
	    } else {
		fprintf(stderr, "No goto at state %d\n", sa->id);
		return 0;
	    }
	}

	printf("\n----------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------\n");	
    } while(0);

    do {
	da = cl->dass[0];
	for (int b=0; b<4; b++) {
	    int v = (ent->dst_addr>>(b*8))&0xff;
	    printf("Find 0x%X in dst addr %d\n", v, b);
	    map<goto_key_t, State*>::iterator ite = da->go.find(v);
	    if (verbose_level > 2)
		_dump_state(da);
	    if (ite != da->go.end()) {
		da = ite->second;
	    } else {
		fprintf(stderr, "No goto at state %d\n", da->id);
		return 0;
	    }
	}

	printf("\n----------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------\n");
    } while(0);

    do {
	sp = cl->spss[0];
	int pv;
	printf("Find 0x%X in src port\n", ent->src_port);
	if (get_eport(ent->src_port) != ent->src_port) {
	    pv = 0;
	} else
	    pv = get_eport(ent->src_port);

	map<goto_key_t, State*>::iterator ite = sp->go.find(pv);
	if (verbose_level > 2)
	    _dump_state(sp);
	if (ite != sp->go.end()) {
	    sp = ite->second;
	} else {
	    fprintf(stderr, "No goto at state %d\n", sp->id);
	    return 0;
	}
	printf("\n----------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------\n");
    } while (0);

    do {
	dp = cl->dpss[0];
	int pv;
	printf("Find 0x%X in dst port\n", ent->dst_port);
	if (get_eport(ent->dst_port) != ent->dst_port) {
	    pv = 0;
	} else
	    pv = get_eport(ent->dst_port);

	map<goto_key_t, State*>::iterator ite = dp->go.find(pv);
	if (verbose_level > 2)
	    _dump_state(dp);
	if (ite != dp->go.end()) {
	    dp = ite->second;
	} else {
	    fprintf(stderr, "No goto at state %d\n", dp->id);
	    return 0;
	}
	printf("\n----------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------\n");
    } while (0);

    do {
	proto = cl->pss[0];
	int pv;
	printf("Find 0x%X in protocol\n", ent->proto);
	if (get_eproto(ent->proto) != ent->proto) {
	    pv = 0;
	} else
	    pv = get_eproto(ent->proto);

	map<goto_key_t, State*>::iterator ite = proto->go.find(pv);
	if (verbose_level > 2)
	    _dump_state(proto);
	if (ite != proto->go.end()) {
	    proto = ite->second;
	} else {
	    fprintf(stderr, "No goto at state %d\n", proto->id);
	    return 0;
	}
	printf("\n----------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------"
	       "------------------\n");
    } while (0);

    if (verbose_level > 1) {
	printf("\nGot Src addr match:\n");
	_dump_state(sa);
	printf("Got Dst addr match:\n");
	_dump_state(da);
	printf("Got Src port match:\n");
	_dump_state(sp);
	printf("Got Dst port match:\n");
	_dump_state(dp);
	printf("Got Protocol match:\n");
	_dump_state(proto);
    }

    State *ss[4];
    ss[0] = da;
    ss[1] = sp;
    ss[2] = dp;
    ss[3] = proto;

    set<ptn_idx_t> *s1 = new set<ptn_idx_t>();
    assert(s1);

    set<ptn_idx_t> *s2 = new set<ptn_idx_t>(), *tmp;
    assert(s2);

    s1->insert(sa->outputs.begin(), sa->outputs.end());
    for (int i=0; i<4; i++) {
	State *cs = ss[i];
	set<ptn_idx_t>::iterator ite = cs->outputs.begin();
	while (ite != cs->outputs.end()) {
	    if (s1->find(*ite) != s1->end()) {
		s2->insert(*ite);
	    }
	    ++ite;
	}
	tmp = s2;
	s2 = s1;
	s1 = tmp;
	s2->clear();
    }

    delete s2;
    return s1;    
}

#include <sys/time.h>
static void
_gen_rand_ptns(g4c_pattern_t *ptns, int n)
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    srandom((unsigned)(tv.tv_usec));

    for (int i=0; i<n; i++) {
	int nbits = random()%5;
	ptns[i].nr_src_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].src_addr = (ptns[i].src_addr<<8)|(random()&0xff);
	nbits = random()%5;
	ptns[i].nr_dst_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].dst_addr = (ptns[i].dst_addr<<8)|(random()&0xff);
	ptns[i].src_port = random()%(PORT_MASK+100);
	ptns[i].dst_port = random()%(PORT_MASK+100);
	ptns[i].proto = random()%(PROTO_MASK+50);
	ptns[i].idx = i;
    }
}

static void
_dump_ptn(g4c_pattern_t *ptn)
{
    printf("Pattern %d, proto 0x%04X, sa 0x%08X/%-2d,"
	   " da 0x%08X/%-2d, sp 0x%04X, dp 0x%04X\n",
	   ptn->idx, ptn->proto, ptn->src_addr, 32-ptn->nr_src_netbits,
	   ptn->dst_addr, 32-ptn->nr_src_netbits, ptn->src_port, ptn->dst_port);
}

#ifdef _G4C_CL_TEST_
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

    ptns[8].src_addr = 0x84639476;
    ptns[8].nr_src_netbits = 32;
    ptns[8].dst_addr = 0x8d65f322;
    ptns[8].nr_dst_netbits = 32;
    ptns[8].src_port = 0x63;
    ptns[8].dst_port = 0x2f;
    ptns[8].proto = 0x5e;

    int ns = _g4c_cl_build(pptns, np, &cl);
    printf(" %d states in total\n", ns);

    int nrfs = 0;
    for (int k=0; k<cl.states.size(); k++)
	if (cl.states[k]->fail)
	    nrfs++;
    printf(" %d failue states\n", nrfs);
	
    set<ptn_idx_t> *rt;
    rt = _match_pattern(ptns+8, &cl);
    if (rt->size() > 0)
	printf("\nFound\n");
    else
	printf("\nFailed\n");

    printf("target:\n");
    _dump_ptn(ptns+8);
    printf("got:\n");
    set<ptn_idx_t>::iterator ite = rt->begin();
    while (ite != rt->end()) {
	_dump_ptn(pptns+(*ite));
	++ite;
    }
    printf("\n\n");

    rt = _match_pattern(pptns+(np/2), &cl);
    if (rt->size() > 0)
	printf("\nFound %d: ", np/2);
    else
	printf("\nFailed %d: ", np/2);

    ite = rt->begin();
    while (ite != rt->end()) {
	printf("%d ", *ite);
	++ite;
    }
    printf("\n");

    delete rt;

//    ns = getchar();   
    
    return 0;
}
#endif
