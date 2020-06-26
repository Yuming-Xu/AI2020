//
//  main.cpp
//  IDAstar
//
//  Created by 徐宇鸣 on 2020/5/13.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//

#include <iostream>
#include "IDAstar.cpp"
#define FILEPATH "/Users/xuyuming/Downloads/AI/AILab1/input3.txt"
#define SAVEPATH "/Users/xuyuming/Downloads/AI/AILab1/result3_ida.txt"
int main(int argc, const char * argv[]) {
    FILE *fp;
    IDAstarState beginState;
    int i;
    int statebit[26];
    char flag = 0;
    fp = fopen(FILEPATH,"r");
    for (i = 0; i < 5; i++) {
        fscanf(fp, "%d,%d,%d,%d,%d\n",&statebit[i*5],&statebit[i*5+1],&statebit[i*5+2],&statebit[i*5+3],&statebit[i*5+4]);
    }
    
    for (i = 0; i < 25; i++) {
        if (statebit[i] == 0&&flag==0) {
            beginState.zeroPos[0] = i;
            flag=1;
        }else if (statebit[i] == 0&&flag==1) {
            beginState.zeroPos[1] = i;
        }
        beginState.stateBit[i] = statebit[i]+65;
    }
    beginState.stateBit[25] = '\0';
    fclose(fp);
    IDAstar idastar = IDAstar();
    if (!idastar.idAstarSearch(beginState)) {
        printf("bug found!\n");
        return 0;
    }
    fp = fopen(SAVEPATH, "w");
    idastar.printResult(fp);
    fclose(fp);
    return 0;
}
