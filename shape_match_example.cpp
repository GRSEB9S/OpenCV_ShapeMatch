// shape_match_example.cpp

#include<opencv2/opencv.hpp>

#include"shape_match.h"

#include<iostream>
#include<conio.h>         // may have to modify this line if not using Windows

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

    double scaleFactor = 0.5;
    std::cout << "starting program . . ." << "\n\n";

    // read in the model image & convert to grayscale, show error and bail if applicable
    cv::Mat imgModel = cv::imread("./images/diamond_model.png");
    if (imgModel.empty()) {
        std::cout << "error: model image not read from file" << "\n\n";
        _getch();
        return(0);
    }
    cv::cvtColor(imgModel, imgModel, cv::COLOR_BGR2GRAY);
    cv::imshow("imgModel", imgModel);

    // read in the scene image & convert to grayscale, show error and bail if applicable
    cv::Mat imgScene = cv::imread("./images/diamond_scene4.png");
    if (imgScene.empty()) {
        std::cout << "error: scene image not read from file" << "\n\n";
        _getch();
        return(0);
    }
    cv::cvtColor(imgScene, imgScene, cv::COLOR_BGR2GRAY);
    cv::imshow("imgScene", imgScene);

    // if the scale factor is not 1.0, resize both the model & scene images accordingly
    if (!checkIfDoublesAreEqual(scaleFactor, 1.0, 0.1)) {
        cv::resize(imgModel, imgModel, cv::Size(), scaleFactor, scaleFactor);
        cv::resize(imgScene, imgScene, cv::Size(), scaleFactor, scaleFactor);
    }

    // get the starting time (tick count) so we can calculate execution time at the end
    double startingTickCount = (double)cv::getTickCount();

    // find the center of mass of the model
    cv::Point modelCenterOfMass = findModelCenterOfMass(imgModel);
    std::cout << "modelCenterOfMass = " << modelCenterOfMass.x << ", " << modelCenterOfMass.y << "\n\n";

    // get the model point offsets
    std::vector<cv::Point> modelPointOffsets = calculateModelPointOffsets(imgModel, modelCenterOfMass);

    // printModelPointOffsetsToStdOut(modelPointOffsets);       // un-comment this line to print model point offsets to the command line
    std::cout << "modelPointOffsets.size() = " << modelPointOffsets.size() << "\n";

    // build the accumulator table
    cv::Mat accumTable = buildAccumTable(imgScene, modelPointOffsets);
    displayAccumTableVisually(accumTable);
    // print AccumTableToStdOut(accumTable);        // un-comment this line to print the accum table to the command line

    // finally we can look for the shape
    runGeneralHough(imgModel, imgScene, accumTable);
    double timeInSeconds = ((double)cv::getTickCount() - startingTickCount) / cv::getTickFrequency();
    std::cout << "program done !! (took " << timeInSeconds << " seconds)" << "\n";

    cv::waitKey();
    return(0);
}

