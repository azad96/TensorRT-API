
#include "interpolation.h"

__global__ void interpolationKernel(int output_channel, int output_height, int output_width, int input_height, int input_width , const float* input, float* output)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if( col<output_width && row<output_height)
    {
        for(int channel_it=0; channel_it<output_channel ;channel_it++)
        {
            float original_x = ((float)col/(output_width-1)) * (input_width-1);
            float original_y = ((float)row/(output_width-1)) * (input_width-1);

            float nw_x = floor(original_x);
            float nw_y = floor(original_y);

            float ne_x = nw_x + 1;
            float ne_y = nw_y;

            float sw_x = nw_x;
            float sw_y = nw_y + 1;

            float se_x = nw_x + 1;
            float se_y = nw_y + 1;

            float nw = (se_x - original_x) * (se_y - original_y);
            float ne = (original_x - sw_x) * (sw_y - original_y);
            float sw = (ne_x - original_x) * (original_y - ne_y);
            float se = (original_x - nw_x) * (original_y - nw_y);

            float bilinear_value = 0;
            if(nw_x<input_width && nw_x>=0 && nw_y<input_height && nw_y>=0)
            {
                float nw_val = input[channel_it*input_height*input_width + int(nw_y*input_width) + int(nw_x)];
                bilinear_value += nw_val * nw;
            }
            if(ne_x<input_width && ne_x>=0 && ne_y<input_height && ne_y>=0 )
            {
                float ne_val = input[channel_it*input_height*input_width + int(ne_y*input_width) + int(ne_x)];
                bilinear_value += ne_val * ne;
            }
            if(sw_x<input_width && sw_x>=0 && sw_y<input_height && sw_y>=0)
            {
                float sw_val = input[channel_it*input_height*input_width + int(sw_y*input_width) + int(sw_x)];
                bilinear_value += sw_val * sw;
            }
            if(se_x<input_width && se_x>=0 && se_y<input_height && se_y>=0)
            {
                float se_val = input[channel_it*input_height*input_width + int(se_y*input_width) + int(se_x)];
                bilinear_value += se_val * se;
            }

            output[channel_it*output_height*output_width + row*output_width +col] =bilinear_value;
        }
    }
}



/*
__global__ void interpolationKernel(int output_channel, int output_height, int output_width, int input_height, int input_width , const float* input, float* output)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;


    int scale_x = int(output_width/input_width);
    int scale_y = int(output_height/input_height);

    if(col<output_width && row<output_height)
    {
        for(int channel_it=0; channel_it<output_channel ;channel_it++)
        {
            float original_x = float((float)col + 0.5)/scale_x - 0.5;
            float original_y = float((float)row + 0.5)/scale_y - 0.5;
            int x1 = int(floor(original_x));
            int x2 = x1 + 1;
            int y1 = int(floor(original_y));
            int y2 = y1 + 1;

            if(x1 >= input_width ) {x1 = input_width -1;}
            if(x1 < 0 ){x1 = 0;}

            if(x2 >= input_width ) {x2 = input_width -1;}
            if(x2 < 0 ){x2 = 0;}

            if(y1 >= input_height ) {y1 = input_height -1;}
            if(y1 < 0 ){y1 = 0;}

            if(y2 >= input_height ) {y2 = input_height -1;}
            if(y2 < 0 ){y2 = 0;}

            int   pos[4][2] = { { x1, y1 },
                                { x1, y2 },
                                { x2,  y1 },
                                { x2,  y2 } };

            float p11 = input[channel_it*input_height*input_width + pos[0][1]*input_width + pos[0][0]];
            float p12 = input[channel_it*input_height*input_width + pos[1][1]*input_width + pos[1][0]];
            float p21 = input[channel_it*input_height*input_width + pos[2][1]*input_width + pos[2][0]];
            float p22 = input[channel_it*input_height*input_width + pos[3][1]*input_width + pos[3][0]];


            float bilinear_value = (p11*(x2-col)*(y2-row) + p12*(x2-col)*(row-y1) + p21*(col-x1)*(y2-row) + p22*(col-x1)*(row-y1));
            bilinear_value /= ((x2-x1)*(y2-y1));
            output[channel_it*output_height*output_width + row*output_width +col] =bilinear_value;
        }
    }
}*/
/*

__global__ void interpolationKernel(int output_channel, int output_height, int output_width, int input_height, int input_width , const float* input, float* output)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    
    if( col<output_width && row<output_height)
    {
        for(int channel_it=0; channel_it<output_channel ;channel_it++)
        {
            float original_x = ((float)col/output_width) * (input_width);
            float original_y = ((float)row/output_height) * (input_height);

            int   pos[4][2] = { { int(floor(original_x)), int(floor(original_y)) },
                                { int( ceil(original_x)), int(floor(original_y)) },
                                { int(floor(original_x)),  int(ceil(original_y)) },
                                { int( ceil(original_x)),  int(ceil(original_y)) } };

            float c0 = input[channel_it*input_height*input_width + pos[0][1]*input_width + pos[0][0]];
            float c1 = input[channel_it*input_height*input_width + pos[1][1]*input_width + pos[1][0]];
            float c2 = input[channel_it*input_height*input_width + pos[2][1]*input_width + pos[2][0]];
            float c3 = input[channel_it*input_height*input_width + pos[3][1]*input_width + pos[3][0]];
            
            float u = original_x - floor(original_x);
            float v = original_y - floor(original_y);

            float s[4] = { (1-u)*(1-v), u*(1-v), (1-u)*v, u*v };
            
            float bilinear_value = c0*s[0] + c1*s[1] + c2*s[2] + c3*s[3]; 
            output[channel_it*output_height*output_width + row*output_width +col] =bilinear_value;
        }
    }
}*/

cudaError_t Interpolation::DoInterpolation(int outputChannel, int outputHeight, int outputWidth, int inputChannel, int inputHeight, int inputWidth,
                                        const float* input, float* output, cudaStream_t* stream)
{
    if(!input || !output)
        return cudaErrorInvalidDevicePointer;
    
    int num_of_elements = outputHeight * outputWidth;

    if(num_of_elements <= 0)
        return cudaErrorInvalidValue;

    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = (outputWidth-1) / block.x +1;
    grid.y = (outputHeight-1) / block.y +1;

    interpolationKernel<<<grid, block, 0, *stream>>>(outputChannel, outputHeight, outputWidth, inputHeight, inputWidth, input, output);
    return CUDA(cudaGetLastError());
}
