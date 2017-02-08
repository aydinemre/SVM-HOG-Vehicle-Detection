#ifndef INC_141044090_EMRE_AYDIN_CREATERAWDATA_H
#define INC_141044090_EMRE_AYDIN_CREATERAWDATA_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>

using namespace std;
using namespace cv;
using namespace ml;

int createRawDataWithConstantSize(string videoName,string path,string extension,
                                  Size rawDataSize,int frameNumber = -1);



#endif //INC_141044090_EMRE_AYDIN_CREATERAWDATA_H
