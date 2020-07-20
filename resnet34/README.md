# How To Run

1.  Execute the gen_wts.py file in pytorch-resnet50 to generate the appropriate weight file with wts extension. This file will be used by TensorRT-API. I used the pretrained resnet50. If you want to load your own weights, use below code segment 

~~~
model = torch.load('resnet50.pth')
model.cuda()
model.eval()
~~~

2. Create a folder named resources in TensorRT-API/resnet50/ and move the resnet50.wts there.

~~~
(path: TensorRT-API/resnet50)
mkdir resources
cd resources
mv ../pytorch-resnet50/resnet50.wts .
~~~

3. Create a folder named build in TensorRT-API/resnet50/ and build the project. You may need to modify the CMakeList.txt to use appropriate cuda directories.

~~~
(path: TensorRT-API/resnet50)
mkdir build
cd build
cmake ..
~~~

4. Compile and run the project. First time you run the project, it will create a TensorRT engine inside resources. This engine will be used for later executions as a cache file. If you make a change in CreateEngine function, you need to delete this engine and create a new one.

~~~
(path: TensorRT-API/resnet50/build)
make
./resnet50
~~~
