//
//  IDAstar.hpp
//  IDAstar
//
//  Created by 徐宇鸣 on 2020/5/13.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//

#ifndef IDAstar_hpp
#define IDAstar_hpp
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <set>
#endif /* IDAstar_hpp */
using namespace std;
struct IDAstarState{
    char stateBit[26];
    char zeroPos[2];
    char g;
    char f;
    bool operator == (const IDAstarState& AS) {
        return (strcmp(stateBit,AS.stateBit)==0);
    }
    bool operator < (const IDAstarState& AS) {
        return (strcmp(stateBit,AS.stateBit)<0);
    }
    bool operator > (const IDAstarState& AS) {
        return (strcmp(stateBit,AS.stateBit)>0);
    }
};
bool operator < (IDAstarState a, IDAstarState b) {
    return (a.f < b.f || (a.f == b.f && a.g < b.g));
}
bool operator > (IDAstarState a, IDAstarState b) {
    return (a.f > b.f || (a.f == b.f && a.g > b.g));
}
struct info
{
    char g;
    char h;
    char times;
    bool instack;
};

class IDAstar {
    IDAstarState _endState;
    char infin;
    char d_limit;
    char next_d_limit;
    map<string,info>openlist;
    vector<IDAstarState>path;
public:
    IDAstar();
    char getManhattn(IDAstarState IDAS);
    char getlinerconflict(IDAstarState IDAS);
    char getH(IDAstarState IDAS);
    bool idAstarSearch(IDAstarState beginState);
    bool idAstarSearch_ID(char times);
    int printResult(FILE* fp);
};
