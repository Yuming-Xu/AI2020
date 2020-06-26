//
//  Astar.cpp
//  AILab1_update
//
//  Created by 徐宇鸣 on 2020/5/10.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//
#include <cstring>
#include <queue>
using namespace std;
const int finalyPos[22] = { 0,0,0,0,0,0,2,1,1,1,1,2,2,2,3,3,3,3,3,4,4,4 };
const int finalxPos[22] = { 0,0,1,2,3,4,0,1,2,3,4,2,3,4,0,1,2,3,4,0,1,2 };
const int dir[4] = {1,-1,-5,5};
//rlud
const int wallx[4] = {4,0,-1,-1};
const int wally[4] = {-1,-1,0,4};
AstarState::AstarState( ) {

}
#pragma mark------Astar-------
Astar::Astar() {

}
void Astar::printResult(FILE* fp,AstarState* state) {
    int i,j;
    if(state->parent!=nullptr) {
        printResult(fp, state->parent);
        if((state->parent)->zeroPos[0] != state->zeroPos[0]) {
            if((state->parent)->zeroPos[0] == state->zeroPos[0]+1) {
                    fprintf(fp,"(%d,r); %d\n",state->stateBit[state->zeroPos[0]+1]-65,state->g);
            } else if((state->parent)->zeroPos[0] == state->zeroPos[0]-1) {
                    fprintf(fp,"(%d,l); %d\n",state->stateBit[state->zeroPos[0]-1]-65,state->g);
            } else if((state->parent)->zeroPos[0] == state->zeroPos[0]+5) {
                    fprintf(fp,"(%d,d); %d\n",state->stateBit[state->zeroPos[0]+5]-65,state->g);
            } else if((state->parent)->zeroPos[0] == state->zeroPos[0]-5) {
                    fprintf(fp,"(%d,u); %d\n",state->stateBit[state->zeroPos[0]-5]-65,state->g);
               }
        } else if((state->parent)->zeroPos[1] != state->zeroPos[1]) {
            if((state->parent)->zeroPos[1] == state->zeroPos[1]+1) {
                    fprintf(fp,"(%d,r); %d\n",state->stateBit[state->zeroPos[1]+1]-65,state->g);
            } else if((state->parent)->zeroPos[1] == state->zeroPos[1]-1) {
                    fprintf(fp,"(%d,l); %d\n",state->stateBit[state->zeroPos[1]-1]-65,state->g);
            } else if((state->parent)->zeroPos[1] == state->zeroPos[1]+5) {
                fprintf(fp,"(%d,d); %d\n",state->stateBit[state->zeroPos[1]+5]-65,state->g);
            } else if((state->parent)->zeroPos[1] == state->zeroPos[1]-5) {
                fprintf(fp,"(%d,u); %d\n",state->stateBit[state->zeroPos[1]-5]-65,state->g);
            }
        }
    }
    fprintf(fp, "+------------------------+\n");
    for (i = 0;i < 5; i++) {
        for (j = 0;j < 5; j++) {
            if (state->stateBit[5*i+j]<65+10) {
                fprintf(fp,"| 0%d ",state->stateBit[5*i+j]-65);
            } else {
                fprintf(fp,"| %d ",state->stateBit[5*i+j]-65);
            }
        }
        fprintf(fp, "|\n");
        fprintf(fp, "+------------------------+\n");
    }
}
void simplePrintResult(AstarState state) {
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

char Astar::getlinerconflict(AstarState AS) {
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

char Astar::getH(AstarState AS) {
    return getManhattn(AS)+getlinerconflict(AS);
}
char Astar::getManhattn(AstarState AS) {
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
AstarState* Astar::AstarSearch(AstarState *beginState) {
    string s;
    AstarState* _curState;
    strcpy(_endState.stateBit,"BCDEFHHIJKGHLMNOPQRSTUVAA");
    s.assign(beginState->stateBit);
    beginState->g = 0;
    beginState->f = getH(*beginState);
    _close_and_half_openlist[s] = 0;
    _openlist.push(*beginState);
    do {
        _curState = new AstarState();
        *_curState = _openlist.top();
        _openlist.pop();
        if (*_curState == _endState) {
            return _curState;
        }
        s.assign(_curState->stateBit);
        auto opit = _close_and_half_openlist.find(s);
        if (opit != _close_and_half_openlist.end()) {
            if(opit->second > _curState->g) {
                opit->second = _curState->g;
            } else if (opit->second < _curState->g) {
                delete _curState;
                _curState = nullptr;
                continue;
            }
        } else {
            _close_and_half_openlist[s] = _curState->g;
        }
        //extend node
        //first move without 7
        //printf("child state:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                //i=0-3,rlud
                int movePlace = _curState->zeroPos[j] + dir[i];
                //printf("move zero:%d,movePlace:%d,dir:%d\n",j,movePlace,dir[i]);
                if (movePlace < 25 && movePlace >=0
                    && _curState->zeroPos[j]%5 != wallx[i]
                    && _curState->zeroPos[j]/5 != wally[i]
                    && _curState->stateBit[movePlace]!=65+7
                    && _curState->stateBit[movePlace]!=65+0) {
                    AstarState childState = *_curState;
                    childState.g = _curState->g+1;
                    childState.stateBit[childState.zeroPos[j]] = childState.stateBit[movePlace];
                    childState.stateBit[movePlace] = 65;
                    childState.zeroPos[j] = movePlace;
                    s.assign(childState.stateBit);
                    auto opit = _close_and_half_openlist.find(s);
                    if (opit != _close_and_half_openlist.end()) {
                        if (opit->second > childState.g) {
                            childState.f = childState.g + getH(childState);
                            if (childState.f <= 57) {
                                childState.parent = _curState;
                                opit->second = childState.g;
                                _openlist.push(childState);
                            }
                        }
                    } else {
                        childState.f = childState.g + getH(childState);
                        if (childState.f <= 57) {
                            childState.parent = _curState;
                            _close_and_half_openlist[s] = childState.g;
                            _openlist.push(childState);
                        }
                    }
                }
            }
        }
        //then move with 7
        for (int i = 0; i < 4; i++) {
            //rlud
            int zeroMove0 = _curState->zeroPos[0] + dir[i];
            int zeroMove1 = _curState->zeroPos[1] + dir[i];
            if (_curState->zeroPos[0] < 25 && _curState->zeroPos[0] >=0 && _curState->zeroPos[0]%5 != wallx[i]
                && _curState->zeroPos[0]/5 != wally[i] && _curState->zeroPos[1] < 25 && _curState->zeroPos[1] >=0 && _curState->zeroPos[1]%5 != wallx[i] && _curState->zeroPos[1]/5 != wally[i] && _curState->stateBit[zeroMove0]==65+7 && _curState->stateBit[zeroMove1]==65+7) {
                //printf("move 7,dir:%d\n",dir[i]);
                AstarState childState = *_curState;
                childState.g = _curState->g+1;
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
                auto opit = _close_and_half_openlist.find(s);
                if (opit != _close_and_half_openlist.end()) {
                    if (opit->second > childState.g) {
                        childState.f = childState.g + getH(childState);
                        if (childState.f <= 57) {
                            childState.parent = _curState;
                            opit->second = childState.g;
                            _openlist.push(childState);
                        }
                    }
                } else {
                    childState.f = childState.g + getH(childState);
                    if (childState.f <= 57) {
                        childState.parent = _curState;
                        _close_and_half_openlist[s] = childState.g;
                        _openlist.push(childState);
                    }
                }
            }
        }
    }while(!_openlist.empty());
    return nullptr;
}