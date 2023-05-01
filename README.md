Download Link: https://assignmentchef.com/product/solved-bme695dl-homework-2
<br>
This homework consists of the following three goals:

<ul>

 <li>To introduce you to ImageNet, the world’s largest database of images for developing image recognition algorithms with deep learning techniques.</li>

 <li>To familiarize you with the data loading facilities provided by Torchvision and for you to customize data loading to your needs. To have you hand-craft the backpropagation of loss by a direct calculation of the gradients of the loss with respect to the learnable parameters for a neural network with an input layer, two hidden layers, and one output layer.</li>

</ul>

A good place to start for this homework is to visit the website

http://image-net.org/explore

in order to become familiar with how the dataset of now over 14 million images is organized. As you will notice, the organization is hierarchical in the form of a tree structure. When you click on a node of the tree in the left window, you will see on the right (if you wait long enough for the thumbnails to download) the images corresponding to that category. It is best to click on the leaf nodes of the tree since the number of images for the non-terminal nodes will, in general, be larger and would take longer for the thumbnails to download. Fig. 1 shows a screenshot of Treemap visualization of the “domestic cat” category at the website listed above.

The URL’s to the ImageNet images are stored in files with names like “n03173929” where “n” is a designator for such files and the number that follows is the

Figure 1: ImageNet Treemap visualization for the “domestic cat” category.

actual identifier for the file. For example, the URL’s to the images for the “domestic cat” category reside in a file named “n02121808”. That begs the question: Who or what is the keeper of the mappings from the symbolic names of the different image categories and the corresponding text files that store the URLs. That mapping resides in a file called

imagenet_class_info.json

If you have not encountered a JSON file before, JSON stands for “JavaScript Object Notation”. It’s purely a text file formatted as a sequence of “attributevalue” pairs that has become popular for several different kinds of data exchange between computers. Shown below is one of the entries in the very large file mentioned above:

“n02121808”: {“img_url_count”: 1831,

“flickr_img_url_count”: 1176,

“class_name”: “domestic cat” }

What this says is that the URLs for the “domestic cat” category are to be found in the ImageNet file named ”n02121808” You will be provided with the imagenet_class_info.json file or you can download it directly from GitHub.

With that as an introduction to ImageNet, the sections that follow outline the required programming steps for each programming task. The class, variable, and method names, etc program-defined attributes are not strict. However, make sure to follow the file naming, input argument names and output file format specifications that are required for the evaluation. You won’t need GPU for completing this homework.

For the training task, your homework will involve training a simple neural network that consists of an input layer, two hidden layers, and one output layer. We will use the matrix <em>w</em><sub>1 </sub>to represent the link weights between the input and the first hidden layer, the matrix <em>w</em><sub>2 </sub>the link weights between the first hidden layer and the second hidden layer, and, finally, the matrix <em>w</em><sub>3 </sub>the link weights between the second hidden layer and the output.

For each hidden layer, we will use the notation <em>hi </em>as the output before the application of the activation function and <em>hi<sub>relu </sub></em>for the output after the activation. So if <em>x </em>is the vector representation of the input data, we have the following relationships in the forward direction:

<table width="209">

 <tbody>

  <tr>

   <td width="51"><em>h</em>1</td>

   <td width="25">=</td>

   <td width="132"><em>x.mm</em>(<em>w</em>1)</td>

  </tr>

  <tr>

   <td width="51"><em>h</em>1<em><sub>relu</sub></em></td>

   <td width="25">=</td>

   <td width="132"><em>h</em>1<em>.clamp</em>(<em>min </em>= 0)</td>

  </tr>

  <tr>

   <td width="51"><em>h</em>2</td>

   <td width="25">=</td>

   <td width="132"><em>h</em>1<em><sub>relu</sub>.mm</em>(<em>w</em>2)</td>

  </tr>

  <tr>

   <td width="51"><em>h</em>2<em><sub>relu</sub></em></td>

   <td width="25">=</td>

   <td width="132"><em>h</em>2<em>.clamp</em>(<em>min </em>= 0)</td>

  </tr>

  <tr>

   <td width="51"><em>y</em><em>pred</em></td>

   <td width="25">=</td>

   <td width="132"><em>h</em>2<em><sub>relu</sub>.mm</em>(<em>w</em>3)</td>

  </tr>

 </tbody>

</table>

where <em>.mm</em>() does for tensors what <em>.dot</em>() does for Numpy’s ndarrays. Basically, <em>mm </em>stands for matrix multiplication. Remember that with tensors, a vector is a one-row tensor. That is, when an n-element vector stored in a tensor, its shape is (1<em>,n</em>). So what you see in the first line, “<em>h</em>1 = <em>x.mm</em>(<em>w</em>1)” amounts to multiplying a matrix <em>w</em>1 with a vector <em>x</em>.

Before listing the tasks, you need to also understand how the loss can be backpropagated and the gradients of loss computed for simple neural networks. The following 3-step logic involved is as follows for the case of MSE loss for the last layer of the neural network. You repeat it backwards for the rest of the network.

<ul>

 <li>The loss at the output layer:</li>

</ul>

<em>L </em>= (<em>y </em>− <em>y</em><em>pred</em>)<em>t</em>(<em>y </em>− <em>y</em><em>pred</em>)

where <em>y </em>is the groundtruth vector and <em>y<sub>pred </sub></em>the predicted vector. • Propagating the loss backwards and calculating the gradient of the loss with respect to the parameters in the link weights involves the following three steps:

<ol>

 <li>Find the gradient of the loss with respect to the link matrix <em>w</em><sub>3 </sub>by:</li>

</ol>

<em>grad</em><em>w</em>3 = <em>h</em>2<em>trelu.mm</em>(2 ∗ <em>y</em><em>error</em>)

<ol start="2">

 <li>Propagate the error to the post-activation point in the hidden layer <em>h</em><sub>2 </sub>by</li>

 <li>Propagate the error past the activation in the layer <em>h</em>2 by</li>

</ol>

<em>h</em>2<em><sub>error</sub></em>[<em>h</em>2 <em>&lt; </em>0] = 0

<h1>1             Recommended Python Packages</h1>

The following are some recommended python packages.

torchvision, torch.utils.data, glob,os, numpy, PIL, argparse, requests , logging, json

Note that the list is not exhaustive.

<h1>2             Programming Tasks</h1>

<h2>2.1           Task1: Scraping and Downsampling ImageNet Subset</h2>

<ol>

 <li>Download the provided json file. You can use the json python package to read this file.</li>

 <li>Create py. 3. Specify the following input arguments</li>

</ol>

<table width="488">

 <tbody>

  <tr>

   <td width="488">…<em>#initial import calls </em>import argparse parser = argparse.ArgumentParser(description=’HW02 Task1’)parser.add_argument(’–subclass_list’, nargs=’*’,type=str, required=True)parser.add_argument(’–images_per_subclass’, type=int, required=True)parser.add_argument(’–data_root’, type=str, required=True)parser.add_argument(’–main_class’,type=str, required=True)parser.add_argument(’–imagenet_info_json’, type=str, required=True)args, args_other = parser.parse_known_args()</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

Now call these user specified input arguments in your code using, <em>e</em>.<em>g</em>., args.images_per_subclass. The python call itself would look like as follows

python hw02_ImageNet_Scrapper.py –subclass_list ‘Siamese cat’ ‘Persian cat’ ‘Burmese cat’ 

–main_class ’cat’ –data_root &lt;imagenet_root&gt;/Train/ 

–imagenet_info_json &lt;path_to_imagenet_class_info.json&gt; –images_per_subclass 200

Note that the arguments in the angular brackets are your system specific paths. The above call should download, downsample and save 200 flickr images for ‘Siamese cat’, ‘Persian cat’, and ‘Burmese cat’ each. The images should be stored in &lt;imagenet_root&gt;/Train/cat folder.

<ol start="4">

 <li>Understand the data-structure of imagenet_class_info.json and how to retrieve the necessary information from the ImageNet dataset. The following is an entry in the given .json file</li>

</ol>

<table width="488">

 <tbody>

  <tr>

   <td width="488">…“n02123597”: {“img_url_count”: 1739,“flickr_img_url_count”: 1434,“class_name”: “Siamese cat”} …</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

You can retrieve the url list corresponding to ‘Siamese cat’ subclass using the unique identifier ‘n02123597’. If you open the following link in your browser, you will see the list of urls corresponding to the images of ‘Siamese cat’. http://www.image-net.org/api/text/imagenet.synset.geturls?wnid= n02123597.

You can use the following call in your python code to retrieve the list.

<table width="488">

 <tbody>

  <tr>

   <td width="488"><em>#the_url contains the required url to obtain the full</em><em>list using an identifier</em><em>#the_list_url = http://www.image-net.org/api/text/ imagenet.synset.geturls?</em><em>wnid=n02123597</em>resp = requests.get(the_list_url) urls = [url.decode(’utf-8’) for url in resp.content. splitlines()]for url in urls:<em># download and downsample the required number of images</em></td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

<ol start="5">

 <li>The following is a function skeleton to download an image from a given url. You’re free to handle the try ..except blocks in your own way.</li>

</ol>

<table width="488">

 <tbody>

  <tr>

   <td width="488"><em>’’’</em><em>Reference:https://github.com/johancc/</em><em>ImageNetDownloader</em><em>’’’ </em>import requests from PIL import Imagefrom requests.exceptions import ConnectionError, ReadTimeout,TooManyRedirects,MissingSchema, InvalidURLdef get_image(img_url, class_folder):if len(img_url) &lt;= 1:<em>#url is useless Do something </em>try:img_resp = requests.get(img_url, timeout = 1)except ConnectionError: <em>#Handle this exception </em>except ReadTimeout: <em>#Handle this exception </em>except TooManyRedirects: <em>#handle exception </em>except MissingSchema: <em>#handle exception </em>except InvalidURL: <em>#handle exception</em>if not ’content-type’ in img_resp.headers:<em>#Missing content. Do something </em>if not ’image’ in img_resp.headers[’content-type’]:<em># The url doesn’t have any image. Do something</em></td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

<table width="488">

 <tbody>

  <tr>

   <td width="488">if (len(img_resp.content) &lt; 1000):<em>#ignore images &lt; 1kb</em>img_name = img_url.split(’/’)[-1] img_name = img_name.split(“?”)[0]if (len(img_name) &lt;= 1):<em>#missing image name </em>if not ’flickr’ in img_url:<em># Missing non-flickr images are difficult to</em><em>handle. Do something.</em>img_file_path = os.path.join(class_folder, img_name)with open(img_file_path, ’wb’) as img_f:img_f.write(img_resp.content)<em>#Resize image to 64×64 </em>im = Image.open(img_file_path)if im.mode != “RGB”:im = im.convert(mode=”RGB”)im_resized = im.resize((64, 64), Image.BOX) <em>#Overwrite original image with downsampled image </em>im_resized.save(img_file_path)</td>

  </tr>

 </tbody>

</table>

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

<ol start="6">

 <li>The desired output from the image scrapper is that you should be able to download 600 (200 × 3) training images for the cat class and 600 training images for the dog</li>

 <li>Follow the following folder structure for saving your training and validation images. &lt;imagenet_root&gt;/Train/cat/, &lt;imagenet_root &gt;/Train/dog/, &lt;imagenet_root&gt;/Val/cat/, &lt;imagenet_root&gt;/</li>

</ol>

Val/dog/. You can use os.path.join(…) and os.mkdir(…) for creating the required folder structure.

<ol start="8">

 <li>After the successful implementation of py, you can download the required training and validation sets for Task2 using the following four command-line calls (in any order).</li>

</ol>

python hw02_ImageNet_Scrapper.py –subclass_list ‘Siamese cat’ ‘Persian cat’ ‘Burmese cat’ 

–main_class ‘cat’ –data_root &lt;imagenet_root&gt;/Train/ 

–imagenet_info_json &lt;path_to_imagenet_class_info.json&gt; –images_per_subclass 200

python hw02_imagenetScraper.py –subclass_list ‘hunting dog’ ‘sporting dog’ ‘shepherd dog’ 

–main_class ‘dog’ –data_root &lt;imagenet_root&gt;/Train/ 

–imagenet_info_json &lt;path_to_imagenet_class_info.json&gt; –images_per_subclass 200

python hw02_ImageNet_Scrapper.py –subclass_list ‘domestic cat’ ‘alley cat’ 

–main_class ‘cat’ –data_root &lt;imagenet_root&gt;/Val/ 

–imagenet_info_json &lt;path_to_imagenet_class_info.json&gt; –images_per_subclass 100

python hw02_ImageNet_Scrapper.py –subclass_list ‘working dog’ ‘police dog’ 

–main_class ‘dog’ –data_root &lt;imagenet_root&gt;/Val/ 

–imagenet_info_json &lt;path_to_imagenet_class_info.json&gt; –images_per_subclass 100

<h1>3             Task2: Data Loading, Training, and Testing</h1>

<ol>

 <li>Create py</li>

 <li>Use the following argparse arguments</li>

</ol>

<table width="488">

 <tbody>

  <tr>

   <td width="488">import argparse parser = argparse.ArgumentParser(description=’HW02 Task2’)parser.add_argument(’–imagenet_root’, type=str, required=True)parser.add_argument(’–class_list’, nargs=’*’,type=str, required=True)args, args_other = parser.parse_known_args()</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

The argument imagenet_root corresponds to the top folder containing both Train and Val subfolders as created in Task1. The following is an example call to this script

python hw02_imagenet_task2.py –imagenet_root &lt;path_to_imagenet_root&gt; –class_list ‘cat’ ‘dog’

<h2>3.1           Sub Task1: Creating a Customized Dataloader</h2>

Note that you’re free to choose your own program-defined class and variable names. You might find the glob python package useful for retrieving the list of images from a folder. Make sure to use the input arguments and also avoid using any hard-coded initialization in the class methods. All the required class or method variables for completing this task can be derived from the input arguments or should be initialized from the calling routines.

<table width="527">

 <tbody>

  <tr>

   <td width="527">…from torch.utils.data import DataLoader, Dataset class your_dataset_class(Dataset):def __init__(…):<em>’’’</em><em>Make use of the arguments from argparse initialize your program-defined variables</em><em>e.g. image path lists for cat and dog classes you could also maintain label_array</em><em>0     </em><em>— cat</em><em>1     </em><em>— dog</em></td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

<table width="527">

 <tbody>

  <tr>

   <td width="527"><em>Initialize the required transform</em><em>’’’ </em>def __len__(…):<em>’’’ return the total number of images refer pytorch documentation for more details</em><em>’’’ </em>def __getitem__(…):<em>’’’</em><em>Load color image(s), apply necessary data conversion and transformation</em><em>e.g. if an image is loaded in HxWXC (Height X Width</em><em>X Channels) format</em><em>rearrange it in CxHxW format, normalize values from 0</em><em>-255 to 0-1</em><em>and apply the necessary transformation.</em><em>Convert the corresponding label in 1-hot encoding. Return the processed images and labels in 1-hot encoded format</em><em>’’’</em></td>

  </tr>

 </tbody>

</table>

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

After the successful implementation of this class, you can use the following template to create the dataloaders for the training and validation sets.

<table width="527">

 <tbody>

  <tr>

   <td width="527">transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])train_dataset = your_dataset_class(…,transform,…) train_data_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=10, shuffle=True, num_workers=4)val_dataset = your_dataset_class(…,transform,…) val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=4)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

<h2>3.2           Sub Task2: Training</h2>

For this task train the three layer neural network using the code shown below. The code is shown only to give you an idea of how you can structure your program. But it should get you started.

<table width="527">

 <tbody>

  <tr>

   <td width="527">import torch<em>#TODO Follow the recommendations from the lecture notes to ensure reproducible results</em>dtype = torch.float64device = torch.device(“cuda:0” if torch.cuda.is_available() else “cpu”)epochs = 40 <em>#feel free to adjust this parameter </em>D_in, H1, H2, D_out = 3*64*64, 1000, 256, 2 w1 = torch.randn(D_in, H1, device=device, dtype=dtype) w2 = torch.randn(H1, H2, device=device, dtype=dtype) w3 = torch.randn(H2, D_out, device=device, dtype=dtype) learning_rate = 1e-9 for t in range(epochs):for i, data in enumerate(train_data_loader):inputs, labels = data inputs = inputs.to(device) labels = labels.to(device) x = inputs.view(x.size(0), -1)h1 = x.mm(w1)                                                                                         <em>## In</em><em>numpy, you would say                         h1 = x</em><em>.dot(w1)</em>h1_relu = h1.clamp(min=0) h2 = h1_relu.mm(w2) h2_relu = h2.clamp(min=0) y_pred = h2_relu.mm(w3) <em># Compute and print loss</em>loss = (y_pred – y).pow(2).sum().item() y_error = y_pred – y<em>#TODO : Accumulate loss for printing per epoch </em>grad_w3 = h2_relu.t().mm(2 * y_error) <em>#&lt;&lt;&lt;&lt;&lt;&lt;</em><em>Gradient of Loss w.r.t w3</em>h2_error = 2.0 * y_error.mm(w3.t()) <em># backpropagated error to the h2</em><em>hidden layer</em>h2_error[h &lt; 0] = 0                                                                       <em># We set</em><em>those elements of the backpropagated error</em>grad_w2 = h1_relu.t().mm(2 * h2_error) <em>#&lt;&lt;&lt;&lt;&lt;&lt;</em></td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

<table width="527">

 <tbody>

  <tr>

   <td width="527"><em>Gradient of Loss w.r.t w2</em>h1_error = 2.0 * h2_error.mm(w2.t()) <em># backpropagated error to the h1</em><em>hidden layer</em>h1_error[h &lt; 0] = 0                                                                       <em># We set</em><em>those elements of the backpropagated error</em>grad_w1 = x.t().mm(2 * h1_error)                                      <em>#&lt;&lt;&lt;&lt;&lt;&lt;</em><em>Gradient of Loss w.r.t w2</em><em># Update weights using gradient descent </em>w1 -= learning_rate * grad_w1 w2 -= learning_rate * grad_w2 w3 -= learning_rate * grad_w3<em>#print loss per epoch </em>print(’Epoch %d:t %0.4f’%(t, epoch_loss))<em>#Store layer weights in pickle file format </em>torch.save({’w1’:w1,’w2’:w2,’w3’:w3},’./wts.pkl’)</td>

  </tr>

 </tbody>

</table>

37

38

39

40

41

42

43

44

45

46

47

48

49

50

<h2>3.3           Sub Task3: Testing on the Validation Set</h2>

Adapt the incomplete code template from the previous section to load the saved weights and evaluate on the validation set. Print the validation loss and the classification accuracy.

<h1>4             Output Format</h1>

Store your training and validation results in output.txt file, in the following format.

<table width="527">

 <tbody>

  <tr>

   <td width="527">Epoch 0: epoch0_lossEpoch 1: epoch1_lossEpoch 2: epoch2_loss...Epoch n: epochn_loss&lt;blank line&gt;Val Loss: val_lossVal Accuracy: val_accuracy_value%</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10