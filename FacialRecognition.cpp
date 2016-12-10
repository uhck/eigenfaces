#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigenfaces.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

static void read_csv(const string filename, vector<Mat>& images, vector<string>& names, char separator = ',');

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << "<input file> <output folder>\n";
        exit(1);
    }
    
    vector<Mat> images;
    vector<string> names;
    read_csv(string(argv[1]), images, names);
    
    cout << "Read images in: # of images = " << images.size() << "| # of names = " << names.size() << endl << endl;//debug
    
    Eigenfaces faces(images, names);
    cout << "Eigenfaces object created.\n\n"; //debug

    return 0;
}


static void read_csv(const string filename, vector<Mat>& images, vector<string>& names, char separator)
{
    ifstream file(filename.c_str(), ifstream::in);
    if (!file)
    {
        string err_msg = "Error: Invalid input file.\n";
        cout << err_msg << endl;
        exit(1);
    }

    string line, path, imglabel;

    while (getline(file, line))
    {
        stringstream ss(line);
        //Reads path to image, ",", and the image label from stringstream
        getline(ss, path, separator);
        getline(ss, imglabel);
        //If path and image label exists,
        if (!path.empty() && !imglabel.empty())
        {
            //Reads image in grayscale and adds to the vector
            images.push_back(imread(path, 0));
            names.push_back(imglabel);
        }
    }
}

