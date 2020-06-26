//
//  main.cpp
//  test
//
//  Created by 徐宇鸣 on 2020/5/9.
//  Copyright © 2020 徐宇鸣. All rights reserved.
//

#include <set>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
set<string>testlist;
int main(int argc, const char * argv[]) {
    string test = "abcdefg";
    testlist.insert(test);
    set<string>::iterator it;
    it = testlist.find(test);
    if (it!=testlist.end()) {
        cout<<"true"<<endl;
    }
    return 0;
}
