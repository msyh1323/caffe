#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat convolution(Mat image, Mat conv_image, Mat FF, int S, int F, int W_row, int W_column, int P, int C_rows, int C_cols, int CH, double h);
Mat Max_Pooling(Mat image, Mat max_image, int W_rows, int W_cols, int F, int S, int CH);
Mat activation(Mat sigma,int row,int column,int CH);


int main(){
    Mat image;
    
    image = imread("cat.jpg", IMREAD_COLOR);
    int W_rows = image.rows;
    int W_cols = image.cols;
    
    if(image.empty())
	{
		cout << "Could not open of find the image" << endl;
		return -1;
	}
    double **FF;
    int S,F,P,CH;  

    printf("Channel : "); scanf("%d",&CH);
    printf("F size : "); scanf("%d",&F);
    printf("Padding : "); scanf("%d",&P);
    printf("Stride : "); scanf("%d",&S);
    printf("\n");
    

    FF = (double**)malloc(F*sizeof(double*));
    for(int i=0; i<F; i++){
        *(FF+i) = (double*)malloc(F*sizeof(double));
    }

    int C_rows = ((W_rows-F+2*P)/S)+1;
    int C_cols = ((W_cols-F+2*P)/S)+1;
    Mat conv_image(C_rows,C_cols,image.type());
    Mat max_image((W_rows-F)/S+1,(W_cols-F)/S+1,image.type());
    
    int check;
    printf("1.convolution    2.max_pooling  ");
    scanf("%d",&check);

    if(check == 1){
        
	            
        double sum=0;
        int s=0;

        
        Mat kernel;
        // kernel = (Mat_<double>(5,5)<<1,  4,  7,  4,  1,
        //                         4,  16, 26, 16, 4,
        //                         7,  26, 41, 26, 7,
        //                         4,  16, 26, 16, 4,
        //                         1,  4,  7,  4,  1);
                
        kernel = (Mat_<double>(3,3)<<1,2,1,2,4,2,1,2,1);
        //kernel = (Mat_<double>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);

        
        for(int i=0; i<F; i++){
            for(int j=0; j<F; j++){
                sum += kernel.at<double>(i,j);
            }
        }
        if(sum>1){
            for(int i=0; i<F; i++){
                for(int j=0; j<F; j++){
                    kernel.at<double>(i,j) = kernel.at<double>(i,j)/sum;
                }
            }
        }
        
        convolution(image,conv_image,kernel,S,F, W_rows, W_cols,P,C_rows,C_cols,CH,sum);

        Mat dst(C_rows,C_cols,image.type());
        
        filter2D(image, dst, -1, kernel, Point(-1, -1), 0, 0);
        
        
        
	    Mat output = conv_image-dst;


        for(int i=0; i<C_rows; i++){
            for(int j=0; j<C_cols; j++){
                for(int k=0; k<CH; k++){
                    if(conv_image.at<cv::Vec3b>(i,j)[k]!=dst.at<cv::Vec3b>(i,j)[k]) s++;
                }
            }
        }
        
        // for(int i=200; i<203; i++){
        //     for(int j=200; j<203; j++){
        //         printf("%d %d %d  ",image.at<cv::Vec3b>(i,j)[0],conv_image.at<cv::Vec3b>(i,j)[0],dst.at<cv::Vec3b>(i,j)[0]);
        //     }
        //     printf("\n");
        // }
        

        printf("\n\n%d\n\n",s);
        imwrite("dst.jpg",dst);
        imwrite("blur.jpg",output);
        imwrite("conv.jpg",conv_image);
        imshow("conv_image",conv_image);
        imshow("dst",dst);
        imshow("output",output);
        waitKey(0);
    }
    else if(check ==2){
        Max_Pooling(image,max_image,W_rows,W_cols,F,S,CH);
        imwrite("max.jpg",max_image);
        imshow("max.jpg",max_image);
        waitKey(0);
    }
    else{printf("Error");}
   
    return 0;
}

Mat convolution(Mat image, Mat conv_image, Mat FF, int S, int F, int W_row, int W_column, int P, int C_rows, int C_cols, int CH, double h){
    int E;
    clock_t start, end;
    start = clock();
    //Mat padding(W_row+2*P,W_column+2*P,image.type());
    double padding[CH][W_row+2*P][W_column+2*P] = {};
    double temp;
    
    
    for(int i=0; i<W_row; i++){
		for(int j=0; j<W_column; j++){
            for(int k=0; k<CH; k++){
			   padding[k][i+P][j+P] = image.at<cv::Vec3b>(i,j)[k];
            }
		}
	}
    
    for(int o=0; o<CH; o++){
        for(int i=0; i<C_rows; i++ ){
            for(int j=0; j<C_cols; j++){
                for(int k=0; k<F; k++){
                    for(int l=0; l<F; l++){
                        temp += padding[o][k+i*S][l+j*S]*FF.at<double>(k,l);
                    }
                }
                if(temp>255){
                    temp = 255;
                }
                if(temp<0){
                    temp = 0;
                }
                temp = floor(temp+0.5);
                conv_image.at<cv::Vec3b>(i,j)[o] = temp;
                
                temp=0;
            }
        }
    }

    end = clock();
    double result = double(end - start);
    printf("%f\n",result);

    return conv_image;
}

Mat Max_Pooling(Mat image, Mat max_image, int W_rows, int W_cols, int F, int S, int CH){
    clock_t start, end;
    start = clock();
	for(int i=0; i<((W_rows-F)/S)+1; i++){
		for(int j=0; j<((W_cols-F)/S)+1; j++){
			for(int k=0; k<F; k++){
				for(int l=0; l<F; l++){
                    for(int o=0; o<CH; o++){
                        if(max_image.at<cv::Vec3b>(i,j)[o] < image.at<cv::Vec3b>(k+i*S,l+j*S)[o]){
						    max_image.at<cv::Vec3b>(i,j)[o] = image.at<cv::Vec3b>(k+i*S,l+j*S)[o];
					    }
                    }
				}
			}
		}
	}
	end = clock();
    double result = double(end - start);
    printf("%f\n",result);

	return max_image;
}

Mat activation(Mat sigma,int row,int column,int CH){  
    for(int i=0; i<row; i++){
		for(int j=0; j<column; j++){
            for(int k=0; k<CH; k++){
			   sigma.at<cv::Vec3b>(i,j)[k] = 1/(1 + exp(- sigma.at<cv::Vec3b>(i,j)[k]) );
            }
		}
	}
    return sigma;
}
