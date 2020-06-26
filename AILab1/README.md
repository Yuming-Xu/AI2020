### 注：个人使用的环境为macos，g++，在其他环境中可能会出现报错

#### 对于Astar/IDAstar

先访问相应的src文件，然后找到main.cpp文件，将

```
#define FILEPATH "/Users/xuyuming/Downloads/AI/AILab1/digit/input/input1.txt"
#define SAVEPATH "/Users/xuyuming/Downloads/AI/AILab1/digit/output/result1_test1.txt"
```

改为实际在电脑上的文件路径以及保存路径

之后

```
g++ -O3 main.cpp
./a.out
```

#### 对于sudoku

先访问相应的src文件，在这里最终版本为**sudoku_rewrite.cpp**，将

```
#define FILEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/input/sudoku01.txt"
#define SAVEPATH "/Users/xuyuming/Downloads/AI/AILab1/sudoku/output/sudoku01_1.txt"
```

改为实际在电脑上的文件路径以及保存路径

之后

```
g++ -O3 sudoku_rewrite.cpp
./a.out
```





