#ifndef __G4C_AC_HH__
#define __G4C_AC_HH__

#include <cstdio>
#include <vector>
#include <queue>
#include <set>
#include <map>

using namespace std;

class ACState;

class ACState {
public:
	int id;
	ACState *prev;
	map<char, int> go;
	set<int> output;
	int failure;

	int transition[128];

	ACState():id(0), prev(0), failure(-1) {}
	ACState(int sid):id(sid), prev(0), failure(-1) {}
	ACState(int sid, ACState *sprev):id(sid), prev(sprev), failure(-1) {}
	~ACState() {}
};

class ACMachine {
public:
        vector<ACState*> states;
	char **patterns;
	int npatterns;

	ACMachine() {}
	~ACMachine() {
		for (int i=0; i<states.size(); i++)
			delete states[i];
		states.clear();
	}
};

// for C
#include "ac.h"

#endif
