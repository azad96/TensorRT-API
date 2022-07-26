# How To Run

1.  Execute the gen_wts.py file in pytorch-hfe to generate the appropriate weight file with wts extension. This file will be used by TensorRT-API. 

~~~
python gen_wts.py --model_weight_file your_weight_file.pth
~~~

* If you don't have a weight file comment the lines below for now.

~~~
state_dict = remove_module_key(torch.load(args.model_weight_file))
model.load_state_dict(state_dict)
~~~

2. Create a folder named resources in TensorRT-API/hfe/ and move the hfe.wts there.

~~~
(path: TensorRT-API/hfe)
mkdir resources
cd resources
mv ../pytorch-hfe/hfe.wts .
~~~

3. Create a folder named build in TensorRT-API/hfe/ and build the project. You may need to modify the CMakeList.txt to use appropriate cuda directories.

~~~
(path: TensorRT-API/hfe)
mkdir build
cd build
cmake ..
~~~

4. Compile and run the project. First time you run the project, it will create a TensorRT engine inside resources. This engine will be used for later executions as a cache file. If you make a change in SerializeEngine method, you need to delete this engine and create a new one.

~~~
(path: TensorRT-API/hfe/build)
make
./hfe
~~~
