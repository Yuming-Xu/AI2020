//
//  Astar.hpp
//  AILab1_update
//
//  Created by 徐宇鸣 on 2020/5/10.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//

#ifndef Astar_hpp
#define Astar_hpp
#include <cstdio>
#include <vector>
#include <iostream>
#include <map>
#include <cstring>
#include <queue>
#endif /* Astar_hpp */
using namespace std;
//状态空间类
struct AstarState{
    char stateBit[26];
    char zeroPos[2];
    char g;
    char f;
    AstarState* parent;
    AstarState();
    bool operator == (const AstarState& AS) {
        return (strcmp(stateBit,AS.stateBit)==0);
    }
};
struct CompStr
{
    bool operator () (AstarState AS1, AstarState AS2) {
        return (AS1.f > AS2.f)||((AS1.f == AS2.f)&&(AS1.g > AS2.g));
    }
};
class Astar {
public:
    priority_queue<AstarState,vector<AstarState>,CompStr>_openlist;
    map<string,char>_close_and_half_openlist;
    AstarState _endState;
    AstarState _curState;
    Astar();
    AstarState* AstarSearch(AstarState* beginState);
    char getManhattn(AstarState IDAS);
    char getlinerconflict(AstarState IDAS);
    char getH(AstarState AS);
    void printResult(FILE* fp,AstarState* state);
};