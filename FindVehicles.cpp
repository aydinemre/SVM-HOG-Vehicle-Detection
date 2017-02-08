#include "Constants.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>

#define IMAGE_RESIZE_FORMAT Size(48,48)
#define VEHICLE_TYPES 4

using namespace std;
using namespace cv;
using namespace ml;

///////////////////////////////////////// Functions Prototypes ///////////////////////////////////////////////////////

/**
 * This function takes a svm trained data file and types of data and loads data.
 * @params svmTrainingFile : SVM trained data file name.
 * @params type : Vehicle type.
 */
void loadAndSetSVMDetector(string svmTrainingFile, int type);

/**
 * Thread function.This function takes an directory name which includes images.
 * For example : ./vehicleType/positive/
 * This function steps:
 * 1) Goes to given path gets all images from directory.
 * 2) Getting hog descriptors from images.
 * 3) Parse pathname
 * 3.a ) Get vehicleType.
 * 3.b ) Get positive or negative
 * 4 ) Insert calculated hog to global parameters.
 * @params pathName : path name.
 */
void *getHogDescriptorFromDirectory(void *pathName);

/**
 * This function takes a path name and output paramter as vector.
 * Takes all images from given path and push to output vector.
 * @params pathName : path name
 * @params outputVector : output vector which will hold images which is in given path.
 */
void getAllImagesFromDirectory(string pathName, vector<Mat> &outputVector);

/**
 * This function takes image list and computing hog descriptors.
 * @params imageList : List of images.
 * @params descriptrList : list of hog descriptors.
 */
void getHogDescriptors(const vector<Mat> &imageList, vector<Mat> &descriptorList);

/**
 * This function takes an descriptor list and push it to global list using flag and type.
 * Thread safe function.
 * @params descriptors : Descriptor list
 * @params flag : Positive or negative
 * @params type : Type of vehicle
 */
void insertHogDescriptorSafe(vector<Mat> descriptors, int flag, int type);

/**
 * This function trains all vehicle types which defined in Constant.h
 */
void trainSVM(void);

/**
 * trainSVM helper function takes a list descriptor list,svm trained file name and
 * vehicle type. This function creates a svm object training saved svm vectors.
 * @params list : Descritor list.
 * @params savedFile: svm file.
 * int i : Type of vehicle.
 */
void trainVehicle(vector<Mat> list, string savedFile, int i);

/**
 * This function getting transpose image. I used function which is in given link
 * @param sample : Sample images
 * @param trainData : output of transpose data
 * @link : https://github.com/ozcanovunc/opencv-samples/tree/master/vehicle-detection-hog
 */
void getTranspose(const std::vector<cv::Mat> &sample, cv::Mat &trainData);

/**
 * Start to process video.
 */
void startVideo();

/**
 * Takes an path name and parsing path then returns vehicle type.
 * Example path : ./Data/Car/Positive/
 * @params path : path name
 * @returns : vehicle type.
 */
int getVehicleType(String path);

/**
 * Takes an path name and parsing path then returns positive or not information.
 * Example path : ./Data/Car/Positive/
 * @params path : path name
 * @returns : Positive flag.
 */
int getPositiveFlag(String name);

/**
 * This function takes a svm object and hog detecor list.
 * @params svm : svm object reference
 * @params hogDetector : hog list.
 * @link : https://github.com/ozcanovunc/opencv-samples/tree/master/vehicle-detection-hog
 */
void getSvmDetector(const Ptr<SVM> &svm, vector<float> &hog_detector);

/**
 * Gets founded location list,color type and vehicle type and draw a rectangle to all list.
 * @params locations : location list which is detected from hog.
 * @params color : color object
 * @params type : type of vehicle.
 */
void draw(vector<Rect> locations, Scalar_<double> color, int type);

/*
 * Thread function.Gets vehicle type and calls hog detect multi scale function.
 * Each type calls this function and each thread detects vehicles from image.
 */
void *detect(void *type);

///////////////////////////////////////// Global variables ///////////////////////////////


// Descriptor list and hogs for each type.
vector<Mat> carDescriptorList;
vector<int> carHog;

vector<Mat> busDescriptorList;
vector<int> busHog;

vector<Mat> truckDescriptorList;
vector<int> truckHog;

vector<Mat> mvDescriptorList;
vector<int> mvHog;

// Video frame, its clone (draw image) and its subimage (copy image)
Mat frame, drawImage, copyFrame;

// Hog descriptor pointer.
HOGDescriptor *hogDescriptor;
int *counter;

// Mutexs and threads ids array.
pthread_mutex_t *lock;
pthread_t *mainThreads;

int main(int argc, char *argv[]) {

    ///////////////////////////////////////////////// Get hog descriptors //////////////////////////////////////////////
    cerr << "Program starting ..." << endl << endl;

    /// Create mutex.
    cerr << endl <<"Creating mutexs ..." << endl;
    lock = new pthread_mutex_t[VEHICLE_TYPES];
    for (int i = 0; i < VEHICLE_TYPES; ++i) {
        if (pthread_mutex_init(&lock[i], NULL) != 0) {
            cerr << "Mutex init failed" << endl;
            return -1;
        }
    }

    cerr << "Do you have svm trained file ? " << endl;
    int choice;
    do {
        cerr << "(1) Yes, i will use old trained file : " << endl;
        cerr << "(2) No, train Data Set" << endl << endl;
        cin >> choice;
    }while(choice != 1 && choice != 2);

    if(choice == 2) {

        // Create thread to get all images from files and create images hog descriptors.
        cerr << "Creating threads ..." << endl;
        pthread_t *threadId;
        threadId = new pthread_t[VEHICLE_TYPES * 2];
        for (int i = 0; i < VEHICLE_TYPES * 2; ++i) {
            cerr << "Getting hog descriptors from this file : " << fileConstants::vehicleDataSetList[i] << endl;
            pthread_create(&threadId[i], NULL, getHogDescriptorFromDirectory,
                           (void *) fileConstants::vehicleDataSetList[i].c_str());
        }

        // Wait for all thread.
        for (int i = 0; i < VEHICLE_TYPES * 2; ++i)
            pthread_join(threadId[i], NULL);
        free(threadId);

        /////////////////////////////////////// Train SVM with HOG Descriptors ////////////////////////////////////////////
        cerr << "Starting to traing Support Vector Machine" << endl;
        trainSVM();
        cerr << "Training finished." << endl;

    }

    /////////////////////////////////////////////////// Process Video /////////////////////////////////////////////////
    cerr << "Creating counter array " << endl;
    counter = new int[VEHICLE_TYPES];
    for (int i = 0; i < VEHICLE_TYPES; ++i)
        counter[i] = 0;

    cerr << "Starting to process video ... " << endl;
    startVideo();
    cerr << "Program closing ... " << endl;

    // Destroy all mutex.
    for (int i = 0; i < VEHICLE_TYPES; ++i)
        pthread_mutex_destroy(&lock[i]);
    free(lock);
    cerr << "Destroyed all mutexs .. " << endl;

    cerr << "Waiting threads ... " << endl;
    for (int i = 0; i < VEHICLE_TYPES; ++i)
        while (pthread_kill(mainThreads[i], 0) != 0);
    cerr << "Finished all threads ... " << endl;

    free(counter);
    free(hogDescriptor);
    return 0;
}

void startVideo() {

    cerr << "Loading trained support vector machine table... " << endl;
    hogDescriptor = new HOGDescriptor[VEHICLE_TYPES];
    for (int i = 0; i < VEHICLE_TYPES; ++i)
        loadAndSetSVMDetector(fileConstants::svmFileList[i], i);

    VideoCapture videoCapture;
    // Open the camera.
    cerr << "Video opening ..." << endl;
    videoCapture.open(fileConstants::INPUT_VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "Unable to open the device" << endl;
        return;
    }

    // Create main threads.
    mainThreads = new pthread_t[VEHICLE_TYPES];
    double fps;
    char key = 0;
    time_t start, end;
    int frameCounter = 0;
    while (key != 'q' && key != 27) {

        if (frameCounter == 0) time(&start);

        videoCapture >> frame;
        resize(frame, frame, Size(640, 640));
        copyFrame = Mat(frame, Rect(0, frame.rows / 2, frame.cols / 2, frame.rows / 2));
        drawImage = frame.clone();

        // Draw a line.
        line(drawImage, Point(frame.cols / 100 * 28, frame.rows / 100 * 60),
             Point(frame.cols / 100 * 50, frame.rows / 100 * 60),
             Scalar(0, 0, 255), 3);

        for (int i = 0; i < VEHICLE_TYPES; ++i) {
            int *arg = (int *) malloc(sizeof(*arg));
            *arg = i;
            pthread_create(&mainThreads[i], NULL, detect, (void *) arg);
        }

        for (int i = 0; i < VEHICLE_TYPES; ++i)
            pthread_join(mainThreads[i], NULL);


        time(&end);
        ++frameCounter;
        double sec = difftime(end, start);
        fps = frameCounter / sec;

        putText(drawImage, "FPS : " + to_string(fps), Point(250, 20), 1, 1, Scalar(0, 0, 255), 2);
        putText(drawImage, "Car : " + to_string(counter[0]), Point(50, 50), 1, 1, Scalar(255, 0, 0), 2);
        putText(drawImage, "Bus : " + to_string(counter[1]), Point(200, 50), 1, 1, Scalar(0, 255, 0), 2);
        putText(drawImage, "Minivan : " + to_string(counter[2]), Point(350, 50), 1, 1, Scalar(0, 0, 255), 2);
        putText(drawImage, "Truck : " + to_string(counter[3]), Point(500, 50), 1, 1, Scalar(0, 0, 0), 2);

        imshow("Windows", drawImage);
        key = (char) waitKey(10);

    }

}

void *detect(void *type) {

    vector<Rect> locations;

    int ind = *((int *) type);
    hogDescriptor[ind].detectMultiScale(copyFrame, locations,0,Size(),Size(),1.10);
    switch (ind) {
        case 0: {
            draw(locations, Scalar(255, 0, 0), 0);
            break;
        }
        case 1: {
            draw(locations, Scalar(0, 255, 0), 1);
            break;
        }
        case 2: {
            draw(locations, Scalar(0, 0, 255), 2);
            break;
        }
        case 3: {
            draw(locations, Scalar(0, 0, 0), 3);
            break;
        }
    }
}

void *getHogDescriptorFromDirectory(void *pathName) {

    String fileName((char *) pathName);

    vector<Mat> imageVector;
    vector<Mat> descriptorListLocal;

    getAllImagesFromDirectory(fileName, imageVector);
    getHogDescriptors(imageVector, descriptorListLocal);

    int positiveFlag = getPositiveFlag(fileName);
    int vehicleType = getVehicleType(fileName);
    insertHogDescriptorSafe(descriptorListLocal, positiveFlag, vehicleType);
}

void getAllImagesFromDirectory(string pathName, vector<Mat> &outputVector) {

    DIR *dirp;
    struct dirent *entry;
    struct stat status;
    Mat tempImage;
    string statFile;
    if ((dirp = opendir(pathName.c_str())) != NULL) {
        while ((entry = readdir(dirp)) != NULL) {
            statFile = pathName + "/" + entry->d_name;
            stat(statFile.c_str(), &status);
            if (S_ISREG(status.st_mode)) {
                tempImage = imread(statFile);
                resize(tempImage, tempImage, IMAGE_RESIZE_FORMAT);
                outputVector.push_back(tempImage);
            }
        }
        closedir(dirp);
    }
}

void getHogDescriptors(const vector<Mat> &imageList, vector<Mat> &descriptorListLocal) {

    HOGDescriptor hogDescriptor;
    hogDescriptor.winSize = IMAGE_RESIZE_FORMAT;
    vector<Point> location;
    vector<float> descriptors;

    for (int i = 0; i < imageList.size(); ++i) {
        Mat grayImage;
        cvtColor(imageList[i], grayImage, COLOR_BGR2GRAY);
        hogDescriptor.compute(grayImage, descriptors, Size(8, 8), Size(0, 0), location);
        descriptorListLocal.push_back(Mat(descriptors).clone());
    }
}

int getVehicleType(String path) {
    size_t found;

    found = path.find("/Cars/");
    if (found != string::npos)
        return 1;

    found = path.find("/Bus/");
    if (found != string::npos)
        return 0;

    found = path.find("/Minivans/");
    if (found != string::npos)
        return 2;

    found = path.find("/Trucks/");
    if (found != string::npos)
        return 3;


    return -1;
}

int getPositiveFlag(String name) {
    String positive("/Positive/");
    size_t found = name.find(positive);
    if (found != string::npos)
        return 1;

    return -1;
}

void insertHogDescriptorSafe(vector<Mat> descriptors, int flag, int type) {

    switch (type) {
        case 0: { // Bus
            pthread_mutex_lock(&lock[0]);
            if (busHog.size() == 0) busHog.assign(descriptors.size(), flag);
            else busHog.insert(busHog.end(), descriptors.size(), flag);
            busDescriptorList.insert(busDescriptorList.end(), descriptors.begin(), descriptors.end());
            pthread_mutex_unlock(&lock[0]);
            break;
        }
        case 1: { // Car
            pthread_mutex_lock(&lock[1]);
            if (carHog.size() == 0) carHog.assign(descriptors.size(), flag);
            else carHog.insert(carHog.end(), descriptors.size(), flag);
            carDescriptorList.insert(carDescriptorList.end(), descriptors.begin(), descriptors.end());
            pthread_mutex_unlock(&lock[1]);
            break;
        }
        case 2: { // minivan
            pthread_mutex_lock(&lock[2]);
            if (mvHog.size() == 0) mvHog.assign(descriptors.size(), flag);
            else mvHog.insert(mvHog.end(), descriptors.size(), flag);
            mvDescriptorList.insert(mvDescriptorList.end(), descriptors.begin(), descriptors.end());
            pthread_mutex_unlock(&lock[2]);
            break;
        }
        case 3: { // truck
            pthread_mutex_lock(&lock[3]);
            if (truckHog.size() == 0) truckHog.assign(descriptors.size(), flag);
            else truckHog.insert(truckHog.end(), descriptors.size(), flag);
            truckDescriptorList.insert(truckDescriptorList.end(), descriptors.begin(), descriptors.end());
            pthread_mutex_unlock(&lock[3]);
            break;
        }
    }
}

void trainSVM(void) {

    trainVehicle(carDescriptorList, fileConstants::svmFileList[0], 0);
    trainVehicle(busDescriptorList, fileConstants::svmFileList[1], 1);
    trainVehicle(mvDescriptorList, fileConstants::svmFileList[2], 2);
    trainVehicle(truckDescriptorList, fileConstants::svmFileList[3], 3);
}

void trainVehicle(vector<Mat> list, string savedFile, int i) {
    Ptr<SVM> svm = SVM::create();
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
    svm->setGamma(0);
    svm->setKernel(SVM::LINEAR);
    svm->setNu(0.6);
    svm->setP(0.1);
    svm->setC(0.008);
    svm->setType(SVM::EPS_SVR);

    Mat trainData;
    getTranspose(list, trainData);
    switch (i) {
        case 0: {
            svm->train(trainData, ROW_SAMPLE, Mat(carHog));
            break;
        }
        case 1: {
            svm->train(trainData, ROW_SAMPLE, Mat(busHog));
            break;
        }
        case 2: {
            svm->train(trainData, ROW_SAMPLE, Mat(mvHog));
            break;
        }
        case 3: {
            svm->train(trainData, ROW_SAMPLE, Mat(truckHog));
            break;
        }
    }
    svm->save(savedFile);

}

void getTranspose(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData) {

    const int rows = (int) train_samples.size();
    const int cols = (int) std::max(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    vector<Mat>::const_iterator itr = train_samples.begin();
    vector<Mat>::const_iterator end = train_samples.end();

    for (int i = 0; itr != end; ++itr, ++i) {
        if (itr->cols == 1) {
            transpose(*(itr), tmp);
            tmp.copyTo(trainData.row(i));
        }
        else if (itr->rows == 1) {
            itr->copyTo(trainData.row(i));
        }
    }
}

void getSvmDetector(const Ptr<SVM> &svm, vector<float> &hog_detector) {

    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
              (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float) -rho;
}

void draw(vector<Rect> locations, Scalar_<double> color, int i) {
    if (!locations.empty()) {

        for (int j = 0; j < locations.size(); ++j) {
            locations[j].y += (frame.rows / 2);
            Point center(locations[j].x + locations[j].width / 2, locations[j].y + locations[j].height / 2);
            if (abs(center.y - drawImage.rows * 3 / 5) < 1)
                counter[i] = counter[i] + 1;
            rectangle(drawImage, locations[j], color, 2);
        }
    }
}

void loadAndSetSVMDetector(string svmTrainingFile, int i) {

    Ptr<SVM> svm;
    hogDescriptor[i].winSize = IMAGE_RESIZE_FORMAT;
    svm = StatModel::load<SVM>(svmTrainingFile);
    vector<float> hogDetector;
    getSvmDetector(svm, hogDetector);
    hogDescriptor[i].setSVMDetector(hogDetector);

}