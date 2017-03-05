 这是一个 “A Neural Algorithm of Artistic Style ” 论文的tensorflow 实现。
代码基于：
	 1. https://github.com/anishathalye/neural-style/
 	 2. www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
 	 
你可以在我的百度云下载到训练好的vgg19：
	链接: https://pan.baidu.com/s/1qXWtbwK 密码: xbq6
	
用以下命令来自己指定用来训练的图片：
		
		python main.py --style_path="Your style image path" -- content_path="Your content image path"
 
 my implement is  based on:
 	 1. https://github.com/anishathalye/neural-style/
 	 2. www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style	 
 And the paper is in the dir named helper.
 
You can download my vgg19  net in my baidu-netdisk:
	link: https://pan.baidu.com/s/1qXWtbwK 
	password: xbq6

You can specified your own picture to train, just use this command:

	python main.py --style_path="Your style image path" -- content_path="Your content image path"

for more options of the command , please look into the FLAGS in main.py 


	