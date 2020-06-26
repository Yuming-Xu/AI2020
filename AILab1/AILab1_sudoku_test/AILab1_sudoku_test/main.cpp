//
//  main.cpp
//  AILab1_sudoku_test
//
//  Created by 徐宇鸣 on 2020/5/15.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//
#include <iostream>
#include <ctime>
#define FILEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/input/sudoku01.txt"
#define SAVEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/output/sudoku01_1.txt"
using namespace std;
class Sudoku {
public:
    int Board[9][9];
    int choice[9][9][10];//zero for remain size;
    int degree[9][9];//for degree h
    int zeroX[81];
    int zeroY[81];
    int zerolen;
    int selUnassignedVal(int* posy, int*posx);
    void initial();
    void update(int y, int x, int input);
    bool forwardChecking(int depth);
};

class SudokuProblem{
public:
    Sudoku result;
    bool backTracking(Sudoku beginState,char dep);
    bool backTracking_rec(Sudoku beginState, char dep);
};

void Sudoku::update(int y, int x, int input) {
    Board[y][x] = input;
    //row
    for (int i = 0; i < 9; i++) {
        if (Board[i][x] == 0) {
            //still zero
            //minus one choice if need
            if (choice[i][x][input] == 1) {
                choice[i][x][0]--;
                choice[i][x][input] = -1;
            }
            //since we only need to fill in blank,
            //every time we just update the zero info
            degree[i][x]--;
        }
    }
    //col
    for (int i = 0; i < 9; i++) {
        if (Board[y][i] == 0) {
            //still zero
            //minus one choice
            if (choice[y][i][input] == 1) {
                choice[y][i][0]--;
                choice[y][i][input] = -1;
            }
            //since we only need to fill in blank,
            //every time we just update the zero info
            degree[y][i]--;
        }
    }
    //block
    //belong to block
    int bcol = y/3*3;
    int brow = x/3*3;
    for (int k = 0; k <9; k++) {
        int tx = bcol + k/3;
        int ty = brow + k%3;
        if ((tx != y) && (ty != x)&&Board[tx][ty] == 0) {
            //need to cut the same col and the same row
            //still zero
            //minus one choice
            if (choice[tx][ty][input] == 1) {
                choice[tx][ty][0]--;
                choice[tx][ty][input] = -1;
            }
            //since we only need to fill in blank,
            //every time we just update the zero info
            degree[tx][ty]--;
        }
    }
    //linear
    if (x == y) {
        for (int i = 0; i < 9; i++) {
            if ((i/3 != x/3)&&Board[i][i] == 0) {
                //need to cut off the same block
                //still zero
                //minus one choice
                if (choice[i][i][input] == 1) {
                    choice[i][i][0]--;
                    choice[i][i][input] = -1;
                }
                //since we only need to fill in blank,
                //every time we just update the zero info
                degree[i][i]--;
            }
        }
    }
    if (x + y == 8) {
        for (int i = 0; i < 9; i++) {
            if ((i/3 != y/3) && Board[i][8-i] == 0) {
                //cut off the same block
                //still zero
                //minus one choice
                if (choice[i][8-i][input] == 1) {
                    choice[i][8-i][0]--;
                    choice[i][8-i][input] = -1;
                }
                //since we only need to fill in blank,
                //every time we just update the zero info
                degree[i][8-i]--;
            }
        }
    }
}

void Sudoku::initial() {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            degree[i][j] = 20;
            //first we all can assign 9 numbers;
            choice[i][j][0] = 9;
            for (int k = 1; k < 10; k++) {
                //1 stands for still can be chosen
                choice[i][j][k] = 1;
            }
        }
    }
    for (int i = 0; i < 9; i++) {
        degree[i][i] += 6;
        degree[i][8-i] += 6;
    }
    zerolen = 0;
}

int Sudoku::selUnassignedVal(int*posy,int*posx) {
    int minIndex = -1;
    int minChoice = 10;
    int maxDegree = 0;
    for (int i = 0; i < zerolen; i++) {
        if (Board[zeroY[i]][zeroX[i]] == 0) {
            if (choice[zeroY[i]][zeroX[i]][0] < minChoice) {
                minChoice = choice[zeroY[i]][zeroX[i]][0];
                minIndex = i;
                maxDegree = degree[zeroY[i]][zeroX[i]];
            } else if (choice[zeroY[i]][zeroX[i]][0] == minChoice && degree[zeroY[i]][zeroX[i]] > maxDegree ) {
                minIndex = i;
                maxDegree = degree[zeroY[i]][zeroX[i]];
            }
        }
    }
    *posx = zeroX[minIndex];
    *posy = zeroY[minIndex];
    return minIndex;
}

bool Sudoku::forwardChecking(int depth) {
    if (depth == 81) return true;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (Board[i][j]==0 && choice[i][j][0] == 0) {
                return false;
            }
        }
    }
    return true;
}

bool SudokuProblem::backTracking(Sudoku beginState,char dep) {
    return backTracking_rec(beginState,dep);
}

long int countNode = 0;

bool SudokuProblem::backTracking_rec(Sudoku beginState,char dep) {
    //printf("current depth:%d\n",dep);
    countNode++;
    if (dep == 81) {
        result = beginState;
        return true;
    }
    int posy,posx;
    beginState.selUnassignedVal(&posy,&posx);
    for (int i = 1; i < 10; i++) {
        if (beginState.choice[posy][posx][i] == 1) {
            //can assign
            Sudoku nextState = beginState;
            nextState.update(posy, posx, i);
            if (nextState.forwardChecking(dep+1)) {
                //printf("assign[%d][%d]:%d\n",posy,posx,i);
                if (backTracking_rec(nextState,dep+1)) {
                    return true;
                }
            }
        }
    }
    return false;
}

int main() {
    int depth = 0;
    FILE* fp;
    int i,j,temp;
    auto begin = clock();
    fp = fopen(FILEPATH, "r");
    Sudoku* beginState = new Sudoku();
    beginState->initial();
    for (i = 0; i < 9; i++) {
        for (j = 0; j < 9; j++) {
            fscanf(fp,"%d",&temp);
            beginState->Board[i][j] = temp;
            if (temp!=0) {
                //printf("now input:[%d][%d]:%d\n",i,j,temp);
                beginState->update(i, j,temp);
                depth++;
            }
            else {
                //record the initial zero nums
                beginState->zeroX[beginState->zerolen] = j;
                beginState->zeroY[beginState->zerolen] = i;
                beginState->zerolen++;
            }
        }
    }
    SudokuProblem* solution = new SudokuProblem();
    if (!solution->backTracking(*beginState, depth)) {
        printf("bug found!\n");
        return 0;
    }
    fp = fopen(SAVEPATH, "w");
    for (i = 0; i < 9; i++) {
        for (j = 0; j < 8; j++) {
            fprintf(fp, "%d ",solution->result.Board[i][j]);
        }
        fprintf(fp, "%d\n",solution->result.Board[i][j]);
    }
    auto end = clock();
    printf("total use: %Lfs\n",(end - begin)/(long double)CLOCKS_PER_SEC);
    printf("total search: %ld nodes\n",countNode);
    return 0;
}
