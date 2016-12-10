#ifndef EIGENFACES_H
#define EIGENFACES_H

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class Eigenfaces {
    public:
        Eigenfaces();
        Eigenfaces(vector<Mat>& imgs, const vector<string>& nms);
        Eigenfaces(Eigenfaces& other);
        ~Eigenfaces();
        const vector<Mat>& getEigenvectors();
        const vector<string>& getNames();
        const Mat& getMeanFace();
        void operator=(Eigenfaces& other);
        void operator+=(Eigenfaces& other);

        friend
        istream &operator>>(istream& in, Eigenfaces &eface);

        friend
        ostream &operator<<(ostream& out, const Eigenfaces &eface);

    private:
        void normalizer(vector<Mat>& images);
        void train(vector<Mat>& images);
        void copy(Eigenfaces& other);
        void nukem();
        void addImage(const Mat& img);
        void addName(const string& nm);
        void addNames(const vector<string>& nms);
        void addEigenvector(const Mat& ev);
        void addEigenvectors(const vector<Mat>& evs);

        vector<Mat> eigenvectors;
        vector<string> names;
        Mat meanface;
};

Eigenfaces::Eigenfaces()
{
    
}

Eigenfaces::Eigenfaces(vector<Mat>& imgs, const vector<string>& nms)
{
    normalizer(imgs);
    names = nms;
    train(imgs);
}

Eigenfaces::Eigenfaces(Eigenfaces& other)
{
    copy(other);
}

Eigenfaces::~Eigenfaces()
{
    nukem();
}

const vector<Mat>& Eigenfaces::getEigenvectors()
{
    return eigenvectors;
}

const vector<string>& Eigenfaces::getNames()
{
    return names;
}

const Mat& Eigenfaces::getMeanFace()
{
    return meanface;
}

void Eigenfaces::addName(const string& nm)
{
    names.push_back(nm);
}

void Eigenfaces::addNames(const vector<string>& nms)
{
    for (size_t i = 0; i < nms.size(); i++)
        names.push_back(nms[i]);
}

void Eigenfaces::addEigenvector(const Mat& ev)
{
    //Train new image to eigenvector
    eigenvectors.push_back(ev);
}

void Eigenfaces::addEigenvectors(const vector<Mat>& evs)
{
    for (size_t i = 0; i < evs.size(); i++)
        eigenvectors.push_back(evs[i]);
}

void Eigenfaces::operator=(Eigenfaces& other)
{
    copy(other);
}

void Eigenfaces::operator+=(Eigenfaces& other)
{
    addEigenvectors(other.getEigenvectors());
    addNames(other.getNames());
}

istream& operator>>(istream& in, Eigenfaces& eface)
{
    string imgfile;
    getline(in, imgfile);
    in.clear();
    Mat image;
    image = imread(imgfile.c_str(), 0);

    if (!image.data)
    {
        cout << "No image data \n";
        return in;
    }

    eface.addName(imgfile);
    return in;
}

ostream& operator<<(ostream &out, const Eigenfaces& eface)
{
    //Output eigenfaces + mean face into folder called "eigenfaces"
    return out;
}

void Eigenfaces::copy(Eigenfaces& other)
{
    eigenvectors = other.getEigenvectors();
    names = other.getNames();
    meanface = other.getMeanFace();
}

void Eigenfaces::normalizer(vector<Mat>& images)
{
    for (int i = 0; i < images.size(); i++)
    {
        Mat regImg = images[i];

        switch(regImg.channels())
        {
            case 1:
                images[i].convertTo(images[i], CV_64FC1, 1./255); //convert back to original scale
                break;
            case 3:
                images[i].convertTo(images[i], CV_64FC1, 1./255);
                break;
            default:
                break;
        }
    }
    //Detect face and crop to align facial features (eyes, nose, mouth)
}

void Eigenfaces::train(vector<Mat>& images)
{

    cout << "Training data set\n\n"; //debug
    Mat image(images[0].size(), images[0].type());
    meanface = Mat(images[0].rows, images[0].cols, images[0].type());
    cout << "Loading image vectors\n\n"; //debug

    for (int i = 0; i < images.size(); i++)
    {
        //Reduce image matrices to vectors (N^2 x 1)
        image = images[i];
        image.reshape(0, images[i].rows*images[i].cols);
        eigenvectors.push_back(image);
        //Add faces to meanface
        meanface += eigenvectors[i];
        //imshow("Before meanface", images[i]); //display
        //waitKey(0);
    }

    cout << "Calculating mean face\n\n"; //debug
    cout << "Image vector: " << eigenvectors[0].rows << " " << eigenvectors[0].cols << endl << endl; //debug
    
    //Calc meanface
    meanface = meanface / eigenvectors.size();
    
    cout << "Subtracting mean face from images\n\n"; //debug
    Mat A(eigenvectors[0].size(), eigenvectors[0].type());
    meanface.reshape(0, meanface.rows); //change back to original size
    imshow("After meanface", meanface); //display
    waitKey(0);

    //Features = image vectors - mean face
    //A = [all features] (each image is a col)
    for (int i = 0; i < eigenvectors.size(); i++)
    {
        eigenvectors[i] -= meanface;
        hconcat(A, eigenvectors[i], A);
    }

    cout << "A: " << A.rows << " " << A.cols << endl << endl;
    cout << "Calculating covariance matrix\n\n"; //debug
    
    //Calculate A^T*A
    Mat C = A.t()*A;
    Mat v, eval;

    cout << "Calculating eigenvectors v\n\n"; //debug
    //Calculate eigenvectors v of T*A
    eigen(C, eval, v);

    cout << "Calculating M largest eigenvectors u\n\n"; //debug
    //Calculate M largest eigenvectors u
    Mat u = A*v;
    
    //Clear eigenvectors vector
    eigenvectors.clear();
    
    cout << "Load eigenvectors into vector & convert to unit vectors\n\n";
    double unit;
    //Transfer u vectors into eigenvectors
    //Convert u vectors into unit vectors
    for (int i = 0; i < u.cols; i++)
    {
        //Calculates magnitude of vector
        unit = norm(u.col(i));
        //Divide every element by unit to get normalize to unit vector
        eigenvectors.push_back(u.col(i)/unit);
    }

    //Resize eigenvectors back to original image dimensions and output first one ; debug
    eigenvectors[0].reshape(0, images[0].rows);
    cout << "Eigenvector size: " << eigenvectors[0].rows << " " << eigenvectors[0].cols << endl << endl; //debug
    imshow("Eigenface", eigenvectors[0]);

    waitKey(0);
}

void Eigenfaces::nukem()
{
    eigenvectors.clear();
    names.clear();
    meanface.release();
}

#endif // EIGENFACES_H

