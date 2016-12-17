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

        //SHOULD NORMALIZE BY UNIT VECTOR
        switch(regImg.channels())
        {
            case 1:
                images[i].convertTo(images[i], CV_64FC1, 1./255);
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
    int IMGAREA = images[0].rows*images[0].cols;
    int HEIGHT = images[0].rows;
    int WIDTH = images[0].cols;
    int IMGTYPE = images[0].type();
    Size IMGSIZE = images[0].size();
    Size VECSIZE = Size(1, IMGAREA);

    cout << "Training data set\n\n"; 
    cout << "Image size: " << images[0].size() << endl << endl;
    meanface = Mat(IMGAREA, 1, IMGTYPE);
    cout << "Loading image vectors\n\n"; 

    for (int i = 0; i < images.size(); i++)
    {
        //Reduce image matrices to vectors (N^2 x 1)
        images[i]= images[i].reshape(0, IMGAREA);
        //Add faces to meanface
        meanface += images[i]; 
    }

    cout << "Calculating mean face\n\n"; 
    meanface = meanface / images.size();
    /*
     * DISPLAY MEANFACE */
    meanface = meanface.reshape(0, HEIGHT);
    imshow("Meanface", meanface);
    meanface = meanface.reshape(0, IMGAREA);
    waitKey(0);
    //*/

    cout << "Subtracting mean face from images\n\n"; 
    Mat A(meanface.size(), IMGTYPE);

    //Features = image vectors - mean face
    //A = [all features] (each image is a col)
    for (int i = 0; i < images.size(); i++)
    {
        images[i] -= meanface;
        hconcat(A, images[i], A);
        /*
         * DISPLAY FEATURES-MEANFACE
        eigenvectors[i] = eigenvectors[i].reshape(0, images[i].rows); //original size
        imshow("Eigenvectors - meanface", eigenvectors[i]); //display
        waitKey(0);
        eigenvectors[i] = eigenvectors[i].reshape(0, images[i].rows*images[i].cols);
        */
    }
    A = A.colRange(1, A.cols);
    cout << "A: " << A.rows << " " << A.cols << endl << endl; //debug

    cout << "Calculating covariance matrix\n\n"; 
    
    //Calculate A^T*A
    Mat C = A.t()*A;
    Mat v, eval;

    cout << "Calculating eigenvectors v\n\n"; 
    //Calculate eigenvectors v of T*A
    eigen(C, eval, v);
    
    cout << "Calculating M largest eigenvectors u\n\n";
    Mat x, y, MM = Mat(meanface.size(), IMGTYPE);
    eigenvectors.clear();
    for (int i = 0; i < v.cols; i++)
    {
        //Divide every element by unit to get normalize to unit vector
        v.col(i) = A*v.col(i);
        eigenvectors.push_back(v.col(i));
        /*
         * DISPLAY EIGENVECTORS
        eigenvectors[i] = eigenvectors[i].reshape(0, images[0].rows); //original size
        imshow("Eigenface", eigenvectors[i]); //display image
        waitKey(0);
        */
        hconcat(MM, meanface, MM);
    }
    MM = MM.colRange(1, MM.cols);

    cout << "Calculating eigenfaces\n\n";
    y = v * A.t();
    cout << "y " << y.rows << " " << y.cols << endl << endl;

    cout << "Reconstructing faces\n\n";
    x = (v.t() * y).t() + MM;
    cout << "x " << x.rows << " " << x.cols << endl << endl;

    Mat image(IMGSIZE, IMGTYPE);
    //Display reconstructed face
    for (int i = 0; i < x.cols; i++)
    {
        image = Mat(x.rows, 1, x.type());
        x.col(i).copyTo(image); 
        image = image.reshape(0, HEIGHT);
        imshow("Reconstructed face", image);
        waitKey(0);
        image = image.reshape(0, 10304);
    }
}

void Eigenfaces::nukem()
{
    eigenvectors.clear();
    names.clear();
    meanface.release();
}

#endif // EIGENFACES_H

