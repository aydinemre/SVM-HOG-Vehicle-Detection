#include "CreateRawDataConstantSize.h"

int createRawDataWithConstantSize(string videoName,string path,string extension,
                                  Size rawDataSize,int frameNumber){
    // Define video capture.
    VideoCapture videoCapture;
    videoCapture.open(videoName);
    if (!videoCapture.isOpened()){
        cerr << "Unable to open the video" << endl;
        return -1;
    }

    Mat frame;
    int imgCounter = 0;
    bool flag;
    for (int i = 0;
         i != frameNumber && videoCapture.read(frame) != false;
         ++i){
        cerr << i << endl;
        for (int j = 0; j + rawDataSize.height < frame.rows; j += rawDataSize.height) {
            for (int k = 0 ; k + rawDataSize.width < frame.cols; k += rawDataSize.width){
                Mat data = frame(Rect(k,j, rawDataSize.height, rawDataSize.width));
                flag = imwrite(path + to_string(imgCounter) + extension, data);
                if (!flag) {
                    cerr << "Create raw data failed" << endl;
                    return -1;
                }
                ++imgCounter;
            }
        }
    }
    return 1;
}

