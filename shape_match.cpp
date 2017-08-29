// shape_match.cpp

#include"shape_match.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfDoublesAreEqual(double dblA, double dblB, double tolearance) {
    if (std::fabs(dblA - dblB) < tolearance) {
        return true;
    } else {
        return false;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Point findModelCenterOfMass(cv::Mat imgModel) {
    // use the Hu Moments center of mass as the model reference points

    // Otsu thresh to a binary image
    cv::Mat imgThresh;
    cv::threshold(imgModel, imgThresh, 0.0, 255.0, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

    // then find outer contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgThresh.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    //un-comment these to show the threshold and contour images
    // cv::Mat imgContours;
    // cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, 2);
    // cv::imshow("imgThresh", imgThresh);
    // cv::imshow("imgContours", imgContours);

    // can use either the first contour or the binary image to get Hu Moments
    cv::Moments moments = cv::moments(contours[0]);
    // cv::Moments moments = cv::moments(imgThresh);

    // using Hu Moments, find & return the center of mass
    int centerX = (int)(moments.m10 / moments.m00);
    int centerY = (int)(moments.m01 / moments.m00);
    return(cv::Point(centerX, centerY));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<cv::Point> calculateModelPointOffsets(cv::Mat imgModel, cv::Point modelCenterOfMass) {
    // for the model image, get Canny edges
    cv::Mat imgModelCannyEdges;
    cv::Canny(imgModel, imgModelCannyEdges, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD);
    cv::imshow("imgModelCannyEdges", imgModelCannyEdges);

    // declare the model point offsets vector
    std::vector<cv::Point> modelPointOffsets;

    // for each pixel in the model Canny edges image . . .
    for (int currentX = 0; currentX < imgModelCannyEdges.cols; currentX++) {
        for (int currentY = 0; currentY < imgModelCannyEdges.rows; currentY++) {
            // if the current pixel value is > 0 . . .
            if (imgModelCannyEdges.at<uchar>(currentY, currentX) > 0) {
                // declare a point that is the difference between (i.e. the offset from) the model center of mass and the non-zero current pixel location, then append to the vector to be returned
                cv::Point pointToAppend(currentX - modelCenterOfMass.x, currentY - modelCenterOfMass.y);
                modelPointOffsets.push_back(pointToAppend);
            }
        }
    }
    return(modelPointOffsets);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat buildAccumTable(cv::Mat imgScene, std::vector<cv::Point> modelPointOffsets) {
    // for the scene image, get Canny edges
    cv::Mat imgSceneCannyEdges;
    cv::Canny(imgScene, imgSceneCannyEdges, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD);
    cv::imshow("imgSceneCannyEdges", imgSceneCannyEdges);

    // declare the accumulator table, a Mat (2D array) the size of the scene image, indexed as accumTable.at<uchar>(y, x)
    cv::Mat accumTable(imgScene.rows, imgScene.cols, CV_16UC1);
    accumTable = cv::Scalar::all(0);

    // for each pixel in the Canny edges image . . .
    for (int currentX = 0; currentX < imgScene.cols; currentX++) {
        for (int currentY = 0; currentY < imgScene.rows; currentY++) {

            /* The following is some trickery to speed up the algorithm greatly.  We could do the following:
            For each point in the image, consider it could be a possible center of mass of the found shape,
            so check for a match to every shape offset point, and this way we would have to iterate through
            every x, y value, and within that, every point in the shape offset list (this would be O(n^3)).
            Instead, we can iterate through every x, y position, check if the scene pixel is non-zero,
            then iterate through the shape offset points list, and work backward from the current x, y by
            subracting each model point offset, this makes the algorithm O(n^2) */

            // if the scene pixel value is > 0 . . .
            if (imgSceneCannyEdges.at<uchar>(currentY, currentX) > 0) {
                // for each point in the R-list
                for (auto &modelPointOffset : modelPointOffsets) {
                    // subtract rather than add b/c we're "going in reverse" from a non-zero point in the shape
                    // back to a possible center of mass
                    int possibleCenterOfMassX = currentX - modelPointOffset.x;
                    int possibleCenterOfMassY = currentY - modelPointOffset.y;
                    // if the possible center of mass is not off the edge of the image in any direction . . .
                    if (possibleCenterOfMassX >= 0 && possibleCenterOfMassX < imgSceneCannyEdges.cols && possibleCenterOfMassY >= 0 && possibleCenterOfMassY < imgSceneCannyEdges.rows) {
                        // using the accumulator x/y as an index, increment that location of the accumulator table by 1
                        accumTable.at<unsigned short>(possibleCenterOfMassY, possibleCenterOfMassX)++;
                    }
                }
            }

            // this code does the same thing, but without the "going in reverse" trick, so it's much slower
            //for (auto &modelPointOffset : modelPointOffsets) {
            //    cv::Point pointToCheck;
            //    pointToCheck.x = currentX + modelPointOffset.x;
            //    pointToCheck.y = currentY + modelPointOffset.y;
            //    if (pointToCheck.x >= 0 && pointToCheck.x < imgSceneCannyEdges.cols && pointToCheck.y >= 0 && pointToCheck.y < imgSceneCannyEdges.rows) {
            //        if (imgSceneCannyEdges.at<uchar>(pointToCheck.y, pointToCheck.x) > 0) {
            //            accumTable.at<unsigned short>(currentY, currentX)++;
            //        }
            //    }
            //}

        }
    }
    return(accumTable);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void runGeneralHough(cv::Mat imgModel, cv::Mat imgScene, cv::Mat accumTable) {
    // get the 5 best matching results in the accum table
    std::vector<AccumTableResult> best5Results = findAccumTableBestNResults(accumTable, 5);
    printAccumTableBestNResultsToStdOut(best5Results);

    // make a copy of the scene image to draw results on
    cv::Mat imgResults = imgScene.clone();

    // convert results image to color so we can draw results in color
    cv::cvtColor(imgResults, imgResults, CV_GRAY2BGR);

    // for each of the best results, draw a small red circle at that location
    for (auto &result : best5Results) {
        cv::circle(imgResults, result.accumTableLocation, 5, SCALAR_RED, -1);
    }

    // get the best result location, then draw a yellow "X" at that location
    AccumTableResult bestResult = best5Results[0];
    drawLetterOnImage(imgResults, "X", bestResult.accumTableLocation, SCALAR_YELLOW);

    // show the results image (same as the scene, but with the red circles and yellow X)
    cv::imshow("imgResults", imgResults);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<AccumTableResult> findAccumTableBestNResults(cv::Mat accumTable, int numBestResults) {
    std::vector<AccumTableResult> accumTableResults;

    // for each entry in the accum table . . .
    for (int currentX = 0; currentX < accumTable.cols; currentX++) {
        for (int currentY = 0; currentY < accumTable.rows; currentY++) {
            // declare and populate an AccumTableResult, then add to the vector
            AccumTableResult accumTableResult;
            accumTableResult.score = accumTable.at<unsigned short>(currentY, currentX);
            accumTableResult.accumTableLocation = cv::Point(currentX, currentY);
            accumTableResults.push_back(accumTableResult);
        }
    }
    // sort the vector of AccumTableResults in descending order based on score (i.e. best score 1st, 2nd best score 2nd, etc.)
    std::sort(accumTableResults.begin(), accumTableResults.end(), AccumTableResult::sortDescendingByScore);
    std::vector<AccumTableResult> bestNResults;       // this will be the return value

    // add the best N results to the vector to return
    for (int i = 0; i < numBestResults; i++) {
        bestNResults.push_back(accumTableResults[i]);
    }
    return bestNResults;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void displayAccumTableVisually(cv::Mat accumTable) {
    unsigned short accumTableMaxScore = 0;

    for (int currentX = 0; currentX < accumTable.cols; currentX++) {
        for (int currentY = 0; currentY < accumTable.rows; currentY++) {
            if (accumTable.at<unsigned short>(currentY, currentX) > accumTableMaxScore) {
                accumTableMaxScore = accumTable.at<unsigned short>(currentY, currentX);
            }
        }
    }
    double scaleFactor = 255.0 / (double)accumTableMaxScore;
    cv::Mat imgAccumTableScaled(accumTable.rows, accumTable.cols, CV_8UC1);

    for (int currentX = 0; currentX < accumTable.cols; currentX++) {
        for (int currentY = 0; currentY < accumTable.rows; currentY++) {
            imgAccumTableScaled.at<uchar>(currentY, currentX) = (uchar)((double)accumTable.at<unsigned short>(currentY, currentX) * scaleFactor);
        }
    }
    cv::imshow("imgAccumTableScaled", imgAccumTableScaled);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void drawLetterOnImage(cv::Mat image, cv::String letter, cv::Point centerPoint, cv::Scalar color) {
    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (double)image.rows / 400.0;       // base font scale on number of rows (i.e. height) of the image
    int intFontThickness = (int)std::round(dblFontScale * 1.5);     // base font thickness on font scale
    int intBaseline = 0;

    // get the text size, then calculate the lower left origin of the text area
    cv::Size textSize = cv::getTextSize(letter, intFontFace, dblFontScale, intFontThickness, &intBaseline);

    cv::Point lowerLeftTextOrigin;

    lowerLeftTextOrigin.x = (int)(centerPoint.x - (textSize.width / 2));
    lowerLeftTextOrigin.y = (int)(centerPoint.y - (textSize.height / 2));

    cv::putText(image, letter, lowerLeftTextOrigin, intFontFace, dblFontScale, color, intFontThickness);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void printModelPointOffsetsToStdOut(std::vector<cv::Point> modelPointOffsets) {
    std::cout << "modelPointOffsets.size() = " << modelPointOffsets.size() << "\n";

    for (int i = 0; i < modelPointOffsets.size(); i++) {
        std::cout << "[" << i << "] = ";
        std::cout << "(" << modelPointOffsets[i].x << ", " << modelPointOffsets[i].y << "), ";
    }
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void printAccumTableToStdOut(cv::Mat accumTable) {
    std::cout << "accumTable.rows = " << accumTable.rows << "\n";
    for (int currentY = 0; currentY < accumTable.rows; currentY++) {
        std::cout << "accumTable row " << currentY << " has " << accumTable.cols << " cols " << "\n";
        for (int currentX = 0; currentX < accumTable.cols; currentX++) {
            std::cout << (int)accumTable.at<unsigned short>(currentY, currentX) << ", ";
        }
        std::cout << "\n";
        // _getch();        // may have to modify this line if not using Windows
    }
    // index the accumulator table at a certain index if desired as follows (remember it's y, x)
    std::cout << "accumTable value at a certain index = " << (int)accumTable.at<unsigned short>(350, 600) << "\n";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void printAccumTableBestNResultsToStdOut(std::vector<AccumTableResult> bestNRestuls) {
    std::cout << "\n" << "Accum Table best " << bestNRestuls.size() << " results: " << "\n";
    std::cout << "score    x  ,     y   " << "\n";
    for (auto &accumTableResult : bestNRestuls) {
        std::cout << accumTableResult.score << "    " << accumTableResult.accumTableLocation.x << ", " << accumTableResult.accumTableLocation.y << "\n";
    }
    std::cout << "\n";
}

