//
// Created by emreaydin on 30.12.2016.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>

#define MAIN_FOLDER "./DataSet/"

using namespace std;
using namespace cv;

void readDirectory(string pathName);

int main(int argc,char* argv[]){

    readDirectory(MAIN_FOLDER);

    return 0;
}

void readDirectory(string pathName){

    DIR *directory;
    struct dirent *entry;
    struct stat status;
    Mat image;
    string top("..");
    string cur(".");
    if ( (directory = opendir(pathName.c_str()) ) != NULL){
        while( (entry = readdir(directory) ) != NULL) {
            cerr << "Entry : " << entry->d_name << endl;
            if (top.compare(entry->d_name) != 0 && cur.compare(entry->d_name) != 0) {
                String statFile(pathName + "/" + entry->d_name);
                stat(statFile.c_str(), &status);
                if (S_ISDIR(status.st_mode)) {
                    cerr << entry->d_name << " is a directory " << endl;
                    readDirectory(statFile);
                }
                else if (S_ISREG(status.st_mode)) {
                    cerr << statFile << " is a file " << endl;
                    image = imread(statFile);
                    resize(image, image, Size(128, 128));
                    imwrite(statFile, image);
                }
            } else {
                cerr << ".. or ." << endl;
            }
            cerr << "End of while : " << pathName << endl;

        }
    } else{
        cerr << "Path file can't open : " << pathName << endl;
    }
}