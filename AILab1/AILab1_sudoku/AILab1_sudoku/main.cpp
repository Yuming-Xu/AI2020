//
//  main.cpp
//  AILab1_sudoku
//
//  Created by 徐宇鸣 on 2020/5/12.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//
#include <iostream>
#define FILEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/input/sudoku03.txt"
#define SAVEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/output/sudoku03.txt"
long count = 0;
class Sudoku {
public:
    //col:0-8,row:9-17,block:18-26,linear:27,28
    char left[29];
    char Board[9][9];
    char remain[9][9];
    char mark[9][9][10];
    char selUnassignedVal();
    void returnVal(char pos);
    bool isConsistent(char pos,char input);
    void assignVal(char pos,char input);
    bool forwardChecking();
    void updateRemain(char x,char y,char input);
};

class SudokuProblem{
public:
    Sudoku result;
    bool backTracking(Sudoku beginState,char dep);
    bool backTracking_rec(Sudoku beginState, char dep);
};

bool Sudoku::isConsistent(char pos, char input) {
    int y = pos/9;
    int x = pos%9;
    if (x == y) {
        for (int i = 0; i < 9; i++) {
            if (Board[i][i] == input) {
                return false;
            }
        }
    }
    if (x+y==8) {
        for (int i = 0; i < 9; i++) {
            if (Board[i][8-i] == input) {
                return false;
            }
        }
    }
    for (int i = 0; i < 9; i++) {
        if (Board[y][i] == input || Board[i][x] == input) {
            return false;
        }
    }
    int boxrow = y/3;
    int boxcol = x/3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if(Board[boxrow*3+i][boxcol*3+j] == input) {
                return false;
            }
        }
    }
    return true;
}

char Sudoku::selUnassignedVal() {
    //ver1: just find the first 0
    /*
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9 ; j++) {
            if (Board[i][j] == 0) {
                return i*9+j;
            }
        }
    }
    return 0;
     */
    //ver2
    int i,j;
    int minIndex = 0;
    for (i = 0; i < 29; i++) {
        if (left[i] == 1) {
            minIndex = i;
            break;
        } else if (left[minIndex]==0&&left[i] > left[minIndex]) {
            minIndex = i;
        } else if (left[i]!=0&&left[i] < left[minIndex]) {
            minIndex = i;
        }
    }
    int minV = 0;
    int remainV = 10;
    if (minIndex < 9) {
        //col most constrains
        for(i = 0; i < 9; i++) {
            if (Board[i][minIndex] == 0 && remain[i][minIndex] < remainV) {
                remainV = remain[i][minIndex];
                minV = 9*i+minIndex;
            }
        }
        return minV;
    } else if (minIndex < 18) {
        //row most
        for(i = 0; i < 9; i++) {
            if (Board[minIndex-9][i] == 0 && remain[minIndex-9][i] < remainV) {
                remainV = remain[minIndex-9][i];
                minV = 9*(minIndex-9)+i;
            }
        }
        return minV;
    } else if (minIndex < 27) {
        //block most
        for(i = 0; i < 3; i++) {
            for(j = 0; j < 3; j++) {
                if (Board[3*((minIndex-18)/3)+i][3*((minIndex-18)%3)+j] == 0
                    && remain[3*((minIndex-18)/3)+i][3*((minIndex-18)%3)+j] < remainV) {
                    remainV = remain[3*((minIndex-18)/3)+i][3*((minIndex-18)%3)+j];
                    minV =  9*(3*((minIndex-18)/3)+i)+3*((minIndex-18)%3)+j;
                }
            }
        }
        return minV;
    } else {
        if (minIndex == 27) {
            for (i = 0; i < 9; i++) {
                if (Board[i][i] == 0 && remain[i][i] < remainV) {
                    remainV = remain[i][i];
                    minV = 10 * i;
                }
            }
            return minV;
        } else {
            for (i = 0; i < 9; i++) {
                if (Board[i][8-i] == 0 && remain[i][8-i] < remainV) {
                    remainV = remain[i][8-i];
                    minV = 8*i+8;
                }
            }
            return minV;
        }
    }
    //printf("no pos can sel!\n");
    return 0;
}

void Sudoku::assignVal(char pos, char input) {
    char x = pos%9;
    char y = pos/9;
    //printf("x:%d,y:%d,input:%d\n",x,y,input);
    Board[y][x] = input;
    left[x]--;
    left[9+y]--;
    left[18+3*(y/3)+x/3]--;
    if (x==y) left[27]--;
    if (x+y==8) left[28]--;
    updateRemain(x,y,input);
    //printf("0,0:%d\n",remain[0][0]);
}


bool SudokuProblem::backTracking(Sudoku beginState,char dep) {
    return backTracking_rec(beginState,dep);
}

bool SudokuProblem::backTracking_rec(Sudoku beginState,char dep) {
    //printf("current depth:%d\n",dep);
    count++;
    if (dep == 81) {
        result = beginState;
        return true;
    }
    char pos = beginState.selUnassignedVal();
    for (int i = 0; i < 9; i++) {
        if (beginState.isConsistent(pos, (char)(i+1))) {
            Sudoku nextState = beginState;
            nextState.assignVal(pos, (char)(i+1));
            if (nextState.forwardChecking()) {
                if (backTracking_rec(nextState,dep+1)) {
                    return true;
                }
            }
        }
    }
    return false;
}
void Sudoku::updateRemain(char x, char y, char input) {
    //col
    for (int i = 0; i < 9; i++) {
        if (Board[i][x] == 0) {
            if (mark[i][x][input] == 0){
                mark[i][x][input] = -1;
                remain[i][x]--;
            }
        }
    }
    //row
    for (int i = 0; i < 9; i++) {
        if (Board[y][i] == 0) {
            if (mark[y][i][input] == 0){
                mark[y][i][input] = -1;
                remain[y][i]--;
            }
        }
    }
    //block
    int bcol = y/3*3;
    int brow = x/3*3;
    for (int k = 0; k < 9; k++) {
        int tx = bcol + k/3;
        int ty = brow + k%3;
        if (Board[tx][ty] == 0) {
            if (mark[tx][ty][input] == 0){
                mark[tx][ty][input] = -1;
                remain[tx][ty]--;
            }
        }
    }
    //two linear
    if (x == y) {
        for (int i = 0; i < 9; i++) {
            if (Board[i][i] == 0)
                if (mark[i][i][input] == 0){
                    mark[i][i][input] = -1;
                    remain[i][i]--;
                }
        }
    }
    if (x == 8-y) {
        for (int i = 0; i < 9; i++) {
            if (Board[i][8-i] == 0)
                if (mark[i][8-i][input] == 0){
                    mark[i][8-i][input] = -1;
                    remain[i][8-i]--;
                }
        }
    }
    //printf("input[%d][%d]:%d\n",y,x,input);
    //printf("remain[1][0]:%d\n",remain[1][0]);
}

bool Sudoku::forwardChecking() {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (Board[i][j] == 0) {
                char temp[9] = {1,2,3,4,5,6,7,8,9};
                if (i==j) {
                    for (int k = 0; k < 9; k++) {
                        if (k!=i && Board[k][k]!=0) {
                            temp[Board[k][k]-1] = -1;
                        }
                    }
                }
                if (i+j==8) {
                    for (int k = 0; k < 9; k++) {
                        if (k!=i && Board[k][8-k]!=0) {
                            temp[Board[k][8-k]-1] = -1;
                        }
                    }
                }
                for (int k = 0; k < 9; k++) {
                    if (k!=j && Board[i][k]!=0) {
                        temp[Board[i][k]-1] = -1;
                    }
                }
                for (int k = 0; k < 9; k++) {
                    if (k!=i && Board[k][j]!=0) {
                        temp[Board[k][j]-1] = -1;
                    }
                }
                int bcol = i/3*3;
                int brow = j/3*3;
                for (int k = 0; k < 9; k++) {
                    int x = bcol + k/3;
                    int y = brow + k%3;
                    if (i!=x && j!=y && Board[x][y]!=0) {
                        temp[Board[x][y]-1] = -1;
                    }
                }
                int flag = 0;
                for (int k = 0; k < 9; k++) {
                    if (temp[k] != -1) {
                        flag = 1;
                    }
                }
                if (flag == 0) {
                    return false;
                }
            }
        }
    }
    return true;
}


int main(int argc, const char * argv[]) {
    auto begin = clock();
    int i,j,k;
    FILE* fp;
    char depth = 0;
    int temp;
    fp = fopen(FILEPATH, "r");
    Sudoku* beginState = new Sudoku();
    for (i = 0; i < 9; i++) {
        for (j = 0; j < 9; j++) {
            beginState->remain[i][j] = 9;
            for (k = 0; k < 9; k++) {
                beginState->mark[i][j][k] = 0;
            }
        }
    }
    for (i = 0; i < 9 ; i++) {
        beginState->left[i] = 9;
        beginState->left[9+i] = 9;
        beginState->left[18+i] = 9;
    }
    beginState->left[27] = 9;
    beginState->left[28] = 9;
    for (i = 0; i < 9; i++) {
        for (j = 0; j < 9; j++) {
            fscanf(fp,"%d",&temp);
            beginState->Board[i][j] = temp;
            if (temp!=0) {
                beginState->updateRemain(j, i,temp);
                beginState->left[j]--;
                beginState->left[9+i]--;
                beginState->left[18+3*(i/3)+j/3]--;
                if (i==j) beginState->left[27]--;
                if (i+j == 8) beginState->left[28]--;
                depth++;
            }
        }
    }
    fclose(fp);
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
    printf("total search: %ld nodes\n",count);
    return 0;
}
