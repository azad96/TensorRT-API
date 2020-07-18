Facebook Multigrain repository: [link](https://github.com/facebookresearch/multigrain)

This sample is an example of using a plugin layer in TensorRT-API.

# How To Run

1.  Execute the gen_wts.py file in pytorch-multigrain to generate the appropriate weight file with wts extension. This file will be used by TensorRT-API. You can download the weights from the multigrain link I shared at the top. If you don't have a weight file comment the lines below for now.

~~~
#checkpoint = torch.load('weight.pth')
#model.load_state_dict(checkpoint['model_state'], strict=False)
~~~

2. Create a folder named resources in TensorRT-API/multigrain/ and move the multigrain.wts there.

~~~
(path: TensorRT-API/multigrain)
mkdir resources
cd resources
mv ../pytorch-multigrain/multigrain.wts .
~~~

3. Create a folder named build in TensorRT-API/multigrain/ and build the project. You may need to modify the CMakeList.txt to use appropriate cuda directories.

~~~
(path: TensorRT-API/multigrain)
mkdir build
cd build
cmake ..
~~~

4. Compile and run the project. First time you run the project, it will create a TensorRT engine inside resources. This engine will be used for later executions as a cache file. If you make a change in CreateEngine function, you need to delete this engine and create a new one.

~~~
(path: TensorRT-API/multigrain/build)
make
./multigrain
~~~
