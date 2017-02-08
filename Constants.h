//
// Created by emreaydin on 30.12.2016.
//

#ifndef INC_141044090_EMRE_AYDIN_DEFINES_H
#define INC_141044090_EMRE_AYDIN_DEFINES_H

#include <string>
#include <map>

using namespace std;

namespace fileConstants{

    string vehicleDataSetList[] ={"./DataSet/Data/Cars/Positive/",
                            "./DataSet/Data/Cars/Negative/",
                            "./DataSet/Data/Minivans/Positive/",
                            "./DataSet/Data/Minivans/Negative/",
                            "./DataSet/Data/Trucks/Positive/",
                            "./DataSet/Data/Trucks/Negative/",
                            "./DataSet/Data/Bus/Positive/",
                            "./DataSet/Data/Bus/Negative/"};

    const string INPUT_VIDEO_FILE = "./Traffic.mp4";
    const string RAW_DATA_PATH = "./DataSet/RawData/";

    string svmFileList[] = {
            "./CAR_SVM",
            "./BUS_SVM",
            "./MV_SVM",
            "./TRUCK_SVM"
    };
}
#endif //INC_141044090_EMRE_AYDIN_DEFINES_H
