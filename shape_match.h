// shape_match.h

#ifndef SHAPE_MATCH_H
#define SHAPE_MATCH_H

#include<opencv2/opencv.hpp>

#include<math.h>

// classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
class AccumTableResult {
public:
    // member variables //////////////////////////////////////////////////////////////////////////////////////////
    int score;
    cv::Point accumTableLocation;
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortDescendingByScore(const AccumTableResult &atrLeft, const AccumTableResult &atrRight) {
        return(atrLeft.score > atrRight.score);
    }
};

// constants /////////////////////////////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

const double MIN_CANNY_THRESHOLD = 90.0;
const double MAX_CANNY_THRESHOLD = 180.0;

// function prototypes ///////////////////////////////////////////////////////////////////////////////////////////
bool checkIfDoublesAreEqual(double dblA, double dblB, double tolerance);
cv::Point findModelCenterOfMass(cv::Mat imgModel);
std::vector<cv::Point> calculateModelPointOffsets(cv::Mat imgModel, cv::Point modelCenterOfMass);
cv::Mat buildAccumTable(cv::Mat imgScene, std::vector<cv::Point> modelPointOffsets);
void runGeneralHough(cv::Mat imgModel, cv::Mat imgScene, cv::Mat accumTable);
std::vector<AccumTableResult> findAccumTableBestNResults(cv::Mat accumTable, int numBestResults);
void displayAccumTableVisually(cv::Mat accumTable);
void drawLetterOnImage(cv::Mat image, cv::String letter, cv::Point centerPoint, cv::Scalar color);
void printModelPointOffsetsToStdOut(std::vector<cv::Point> modelPointOffsets);
void printAccumTableToStdOut(cv::Mat accumTable);
void printAccumTableBestNResultsToStdOut(std::vector<AccumTableResult> bestNResults);

#endif      // end SHAPE_MATCH_H
