//
//  IDAstar.cpp
//  IDAstar
//
//  Created by 徐宇鸣 on 2020/5/13.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//
#include <cstring>
#include <queue>
#include "IDAstar.hpp"
using namespace std;
const int finalyPos[22] = { 0,0,0,0,0,0,2,1,1,1,1,2,2,2,3,3,3,3,3,4,4,4 };
const int finalxPos[22] = { 0,0,1,2,3,4,0,1,2,3,4,2,3,4,0,1,2,3,4,0,1,2 };
const int dir[4] = {1,-1,-5,5};
//rlud
const int wallx[4] = {4,0,-1,-1};
const int wally[4] = {-1,-1,0,4};
void simplePrintResult(IDAstarState state) {
    int i,j;
    printf("+------------------------+\n");
    for (i = 0;i < 5; i++) {
        for (j = 0;j < 5; j++) {
            if (state.stateBit[5*i+j]<65+10) {
                printf("| 0%d ",state.stateBit[5*i+j]-65);
            } else {
                printf("| %d ",state.stateBit[5*i+j]-65);
            }
        }
        printf("|\n");
        printf("+------------------------+\n");
    }
}
#pragma mark------IDAstar-------
int IDAstar::printResult(FILE *fp) {
    int i,j;
    IDAstarState printState = path.back();
    path.pop_back();
    if(!path.empty()) {
        IDAstarState fatherState = path.back();
        printResult(fp);
        if(fatherState.zeroPos[0] != printState.zeroPos[0]) {
            if(fatherState.zeroPos[0] == printState.zeroPos[0]+1) {
                fprintf(fp,"(%d,r); %d\n",printState.stateBit[printState.zeroPos[0]+1]-65,printState.g);
            } else if(fatherState.zeroPos[0] == printState.zeroPos[0]-1) {
                fprintf(fp,"(%d,l); %d\n",printState.stateBit[printState.zeroPos[0]-1]-65,printState.g);
            } else if(fatherState.zeroPos[0] == printState.zeroPos[0]+5) {
                fprintf(fp,"(%d,d); %d\n",printState.stateBit[printState.zeroPos[0]+5]-65,printState.g);
            } else if(fatherState.zeroPos[0] == printState.zeroPos[0]-5) {
                fprintf(fp,"(%d,u); %d\n",printState.stateBit[printState.zeroPos[0]-5]-65,printState.g);
            }
        }else if(fatherState.zeroPos[1] != printState.zeroPos[1]) {
            if(fatherState.zeroPos[1] == printState.zeroPos[1]+1) {
                fprintf(fp,"(%d,r); %d\n",printState.stateBit[printState.zeroPos[1]+1]-65,printState.g);
            } else if(fatherState.zeroPos[1] == printState.zeroPos[1]-1) {
                    fprintf(fp,"(%d,l);%d\n",printState.stateBit[printState.zeroPos[1]-1]-65,printState.g);
            } else if(fatherState.zeroPos[1] == printState.zeroPos[1]+5) {
                fprintf(fp,"(%d,d);%d\n",printState.stateBit[printState.zeroPos[1]+5]-65,printState.g);
            } else if(fatherState.zeroPos[1] == printState.zeroPos[1]-5) {
                fprintf(fp,"(%d,u);%d\n",printState.stateBit[printState.zeroPos[1]-5]-65,printState.g);
            }
        }
    }
    fprintf(fp, "+------------------------+\n");
    for (i = 0;i < 5; i++) {
        for (j = 0;j < 5; j++) {
            if (printState.stateBit[5*i+j]<65+10) {
                fprintf(fp,"| 0%d ",printState.stateBit[5*i+j]-65);
            } else {
                fprintf(fp,"| %d ",printState.stateBit[5*i+j]-65);
            }
        }
        fprintf(fp, "|\n");
        fprintf(fp, "+------------------------+\n");
    }
    return 0;
}

IDAstar::IDAstar(){
    openlist.clear();
    path.clear();
}
char IDAstar::getlinerconflict(IDAstarState AS) {
    int xrow[22] = { 0 }, xcol[22] = { 0 };
    char linearconflict = 0;
    for (int i = 0; i < 25; ++i)
            if (AS.stateBit[i] != 65+0 && AS.stateBit[i] != 65+7) {
                xrow[AS.stateBit[i]-65] = i/5;
                xcol[AS.stateBit[i]-65] = i%5;
            }
            else if (AS.stateBit[i] == 65+7) {
                if ((i%5!=0 && AS.stateBit[i-5] == 65+7) || (i%5 < 4 && AS.stateBit[i+1] == 65+7)) {
                    continue;
                }
                else {
                    xrow[7] = i/5;
                    xcol[7] = i%5;
                }
            }
    for (int row = 0; row < 5; ++row) {
        int temp[5], tot = 0;
        for (int i = 1; i <= 21; ++i)
            if (xrow[i] == row && finalyPos[i] == row) {
                temp[tot++] = i;
            }
        int conf[5] = { 0 };
        int confflag[5][5] = { 0 };
        for (int i = 0; i < tot - 1; ++i) {
            for (int j = i + 1; j < tot; ++j) {
                if (((xcol[temp[i]] > xcol[temp[j]]) && (finalxPos[temp[i]] < finalxPos[temp[j]])) ||
                    ((xcol[temp[i]] < xcol[temp[j]]) && (finalxPos[temp[i]] > finalxPos[temp[j]]))) {
                    ++conf[i];
                    ++conf[j];
                    confflag[i][j] = confflag[j][i] = 1;
                }
            }
        }
        while (1) {
            int maxi = 0;
            bool flag = (conf[maxi] > 0);
            for (int i = 1; i < tot; ++i) if (conf[i] > conf[maxi]) {
                maxi = i;
                flag = true;
            }
            if (!flag) break;
            conf[maxi] = 0;
            for (int i = 0; i < tot; ++i) if (i != maxi)
                if (confflag[i][maxi]) {
                    --conf[i];
                    confflag[i][maxi] = confflag[maxi][i] = 0;
                }
            ++linearconflict;
        }
    }
    for (int col = 0; col < 5; ++col) {
        int temp[5], tot = 0;
        for (int i = 1; i <= 21; ++i)
            if (xcol[i] == col && finalxPos[i] == col) {
                temp[tot++] = i;
            }
        int conf[5] = { 0 };
        int confflag[5][5] = { 0 };
        for (int i = 0; i < tot - 1; ++i) {
            for (int j = i + 1; j < tot; ++j) {
                if (((xrow[temp[i]] > xrow[temp[j]]) && (finalyPos[temp[i]] < finalyPos[temp[j]])) ||
                    ((xrow[temp[i]] < xrow[temp[j]]) && (finalyPos[temp[i]] > finalyPos[temp[j]]))) {
                    ++conf[i];
                    ++conf[j];
                    confflag[i][j] = confflag[j][i] = 1;
                }
            }
        }
        while (1) {
            int maxi = 0;
            bool flag = (conf[maxi] > 0);
            for (int i = 1; i < tot; ++i) if (conf[i] > conf[maxi]) {
                maxi = i;
                flag = true;
            }
            if (!flag) break;
            conf[maxi] = 0;
            for (int i = 0; i < tot; ++i) if (i != maxi)
                if (confflag[i][maxi]) {
                    --conf[i];
                    confflag[i][maxi] = confflag[maxi][i] = 0;
                }
            ++linearconflict;
        }
    }
    return (linearconflict + linearconflict);
}

char IDAstar::getH(IDAstarState AS) {
    return getManhattn(AS)+getlinerconflict(AS);
}
char IDAstar::getManhattn(IDAstarState AS) {
    int i;
    int mkSeven = 0;
    int yCur,xCur,yFin,xFin;
    unsigned char cost = 0;
    for (i = 0; i < 25; i++) {
        if (AS.stateBit[i] != 65+0 && AS.stateBit[i] != 65+7) {
            yCur = i/5;
            xCur = i%5;
            yFin = finalyPos[AS.stateBit[i]-65];
            xFin = finalxPos[AS.stateBit[i]-65];
            cost += (abs(yCur-yFin) + abs(xCur-xFin));
        } else if (AS.stateBit[i] == 65+7) {
            mkSeven = i;
        }
    }
    yCur = mkSeven/5;
    xCur = mkSeven%5;
    yFin = 2;
    xFin = 1;
    cost += (abs(yCur-yFin) + abs(xCur-xFin));
    return cost;
}

bool IDAstar::idAstarSearch_ID(char times) {
    string s;
    IDAstarState _curState = path.back();
    //simplePrintResult(_curState);
    char f = _curState.f;
    if (f > d_limit) {
        next_d_limit = (next_d_limit>f) ? f : next_d_limit;
    } else {
        if (_curState == _endState) {
            return true;
        }
        //extend node
        //first move without 7
        //printf("child state:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                //i=0-3,rlud
                int movePlace = _curState.zeroPos[j] + dir[i];
                //printf("move zero:%d,movePlace:%d,dir:%d\n",j,movePlace,dir[i]);
                if (movePlace < 25 && movePlace >=0
                    && _curState.zeroPos[j]%5 != wallx[i]
                    && _curState.zeroPos[j]/5 != wally[i]
                    && _curState.stateBit[movePlace]!=65+7
                    && _curState.stateBit[movePlace]!=65+0) {
                    IDAstarState childState = _curState;
                    childState.g = _curState.g+1;
                    childState.stateBit[childState.zeroPos[j]] = childState.stateBit[movePlace];
                    childState.stateBit[movePlace] = 65;
                    childState.zeroPos[j] = movePlace;
                    s.assign(childState.stateBit);
                    auto opit = openlist.find(s);
                    if (opit != openlist.end()) {
                        if (opit->second.instack == false) {
                            //not in path
                            childState.f = opit->second.h + childState.g;
                            if (opit->second.times == times) {
                                if (opit->second.g > childState.g) {
                                    opit->second.g = childState.g;
                                    opit->second.instack = true;
                                    path.push_back(childState);
                                    if(idAstarSearch_ID(times)) {
                                        return true;
                                    }
                                    path.pop_back();
                                    opit->second.instack = false;
                                }
                            } else {
                                opit->second.g = childState.g;
                                opit->second.instack = true;
                                opit->second.times = times;
                                path.push_back(childState);
                                if(idAstarSearch_ID(times)) {
                                    return true;
                                }
                                path.pop_back();
                                opit->second.instack = false;
                            }
                        }
                    } else {
                        char h = getH(childState);
                        info tmpinfo = {childState.g,h,times,true};
                        childState.f = h + childState.g;
                        openlist[s] = tmpinfo;
                        path.push_back(childState);
                        if(idAstarSearch_ID(times)) {
                            return true;
                        }
                        path.pop_back();
                        openlist[s].instack = false;
                        //simplePrintResult(childState);
                    }
                }
            }
        }
        //then move with 7
        for (int i = 0; i < 4; i++) {
            //rlud
            int zeroMove0 = _curState.zeroPos[0] + dir[i];
            int zeroMove1 = _curState.zeroPos[1] + dir[i];
            if (_curState.zeroPos[0] < 25 && _curState.zeroPos[0] >=0 && _curState.zeroPos[0]%5 != wallx[i]
                && _curState.zeroPos[0]/5 != wally[i] && _curState.zeroPos[1] < 25 && _curState.zeroPos[1] >=0 && _curState.zeroPos[1]%5 != wallx[i] && _curState.zeroPos[1]/5 != wally[i] && _curState.stateBit[zeroMove0]==65+7 && _curState.stateBit[zeroMove1]==65+7) {
                //printf("move 7,dir:%d\n",dir[i]);
                IDAstarState childState = _curState;
                childState.g = _curState.g+1;
                childState.stateBit[childState.zeroPos[0]] = 65+7;
                childState.stateBit[zeroMove0] = 65;
                childState.zeroPos[0] = zeroMove0;
                childState.stateBit[childState.zeroPos[1]] = 65+7;
                childState.stateBit[zeroMove1] = 65;
                childState.zeroPos[1] = zeroMove1;
                if(childState.stateBit[zeroMove0 + dir[i]]==65+7) {
                    childState.stateBit[zeroMove0] = 65+7;
                    childState.zeroPos[0] = zeroMove0 + dir[i];
                    childState.stateBit[childState.zeroPos[0]] = 65;
                } else {
                    childState.stateBit[zeroMove1] = 65+7;
                    childState.zeroPos[1] = zeroMove1 + dir[i];
                    childState.stateBit[childState.zeroPos[1]] = 65;
                }
                s.assign(childState.stateBit);
                auto opit = openlist.find(s);
                //simplePrintResult(childState);
                if (opit != openlist.end()) {
                    if (opit->second.instack == false) {
                        //not in path
                        childState.f = opit->second.h + childState.g;
                        if (opit->second.times == times) {
                            if (opit->second.g > childState.g) {
                                opit->second.g = childState.g;
                                opit->second.instack = true;
                                path.push_back(childState);
                                if(idAstarSearch_ID(times)) {
                                    return true;
                                }
                                path.pop_back();
                                opit->second.instack = false;
                            }
                        } else {
                            opit->second.g = childState.g;
                            opit->second.instack = true;
                            opit->second.times = times;
                            path.push_back(childState);
                            if(idAstarSearch_ID(times)) {
                                return true;
                            }
                            path.pop_back();

                            opit->second.instack = false;
                        }
                    }
                } else {
                    char h = getH(childState);
                    info tmpinfo = {childState.g,h,times,true};
                    childState.f = h + childState.g;
                    openlist[s] = tmpinfo;
                    path.push_back(childState);
                    if(idAstarSearch_ID(times)) {
                        return true;
                    }
                    path.pop_back();
                    openlist[s].instack = false;
                    //simplePrintResult(childState);
                }
            }
        }
    }
    return false;
}

bool IDAstar::idAstarSearch(IDAstarState beginState) {
    char times = 0;
    string s;
    infin = 100;
    beginState.g = 0;
    info tmpinfo = {0,beginState.f,times,true};
    strcpy(_endState.stateBit,"BCDEFHHIJKGHLMNOPQRSTUVAA");
    d_limit = (beginState.f = getH(beginState));
    s.assign(beginState.stateBit);
    openlist[s] = tmpinfo;
    printf("initial d limit: %d\n",d_limit);
    path.push_back(beginState);
    while (d_limit < infin) {
        times++;
        next_d_limit = infin;
        if (idAstarSearch_ID(times)) {
            return true;
        }
        d_limit = next_d_limit;
        printf("next d limit: %d\n", d_limit);
    }
    return false;
}

