#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>

// Define the weight of the noise in the final blend
// #define NOISE_WEIGHT 0.33 // Adjust this value as needed (e.g., 0.33 for 1/3)

// Define whether to apply low-pass filtering to the noise
#define APPLY_LOW_PASS_FILTER true // Set to false to disable low-pass filtering


// Load a grayscale image
cv::Mat loadImage(const std::string& imageFile) {
    cv::Mat img = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Couldn't open image " << imageFile << ".\n";
        exit(-1);
    }
    return img;
}

// Generate grayscale noise frames
std::vector<cv::Mat> generateNoiseFrames(int width, int height, int numFrames, bool applyFilter) {
    std::vector<cv::Mat> noiseFrames;
    for (int i = 0; i < numFrames; ++i) {
        cv::Mat noise(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = static_cast<float>(rand() % 256);
                noise.at<uchar>(y, x) = static_cast<uchar>(value);
            }
        }
        if (applyFilter) {
            cv::GaussianBlur(noise, noise, cv::Size(5, 5), 0);

            // Transform the noise
            noise.convertTo(noise, CV_32F); // Convert to float for transformation
            noise -= 128; // Subtract 128
            noise *= 2;   // Multiply by 2
            noise += 128; // Add 128 back
            noise.convertTo(noise, CV_8UC1); // Convert back to 8-bit
        }
        noiseFrames.push_back(noise);
    }
    return noiseFrames;
}

// Create a parabolic lookup table
cv::Mat createParabolicLUT() {
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        float normalized = i / 255.0f;
        lut.at<uchar>(i) = static_cast<uchar>(std::round(255.0f * normalized * normalized));
    }
    return lut;
}

// Function to blend images and noise, and apply LUT
void blendImagesAndNoise(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Mat>& noiseFrames, 
                         cv::Mat& resultImg, const cv::Mat& lut, float imageBlendWeight, 
                         float noiseWeight) {
    // Static variable to keep track of the noise frame index
    static int noiseFrameIndex = 0;

    // Use pre-generated noise frame
    cv::Mat noise = noiseFrames[noiseFrameIndex];
    noiseFrameIndex = (noiseFrameIndex + 1) % noiseFrames.size();

    // Blend image1 and image2
    cv::Mat blendedImages;
    cv::addWeighted(img1, imageBlendWeight, img2, imageBlendWeight, 0.0, blendedImages);

    // Blend the result with noise
    cv::Mat finalBlendedImg;
    cv::addWeighted(blendedImages, 1.0f - noiseWeight, noise, noiseWeight, 0.0, finalBlendedImg);

    // Apply the parabolic lookup table using cv::LUT
    cv::LUT(finalBlendedImg, lut, resultImg);
}

int main() {
    std::string imageFile1 = "/home/jim/Desktop/PiTests/images/image.jpg"; // Path to your first image file
    std::string imageFile2 = "/home/jim/Desktop/PiTests/images/image2.jpg"; // Path to your second image file

    cv::Mat img1 = loadImage(imageFile1);
    cv::Mat img2 = loadImage(imageFile2);

    const int numNoiseFrames = 30;
    std::vector<cv::Mat> noiseFrames = generateNoiseFrames(img1.cols, img1.rows, numNoiseFrames, APPLY_LOW_PASS_FILTER);

    cv::namedWindow("Blended Image Playback", cv::WINDOW_NORMAL); // Use WINDOW_NORMAL to allow resizing
    cv::resizeWindow("Blended Image Playback", img1.cols, img1.rows); // Resize the window to the image size

    const float NOISE_WEIGHT = .6;
    // Calculate the weight of the blended image
    const float imageBlendWeight = 1.0f - NOISE_WEIGHT;
    const float noiseWeight = NOISE_WEIGHT;

    // Create the parabolic lookup table
    cv::Mat lut = createParabolicLUT();

    while (true) {
        auto loopStartTime = std::chrono::steady_clock::now();

        cv::Mat transformedImg;
        blendImagesAndNoise(img1, img2, noiseFrames, transformedImg, lut, imageBlendWeight, noiseWeight);

        transformedImg *= 1.5;

        cv::imshow("Blended Image Playback", transformedImg);

        // Check for escape key press
        int key = cv::waitKey(1);
        if (key == 27) { // ASCII code for the escape key
            break;
        }

        auto loopEndTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = loopEndTime - loopStartTime;
        std::cout << "Loop duration: " << elapsed_seconds.count() << "s\n";
    }

    return 0;
}
