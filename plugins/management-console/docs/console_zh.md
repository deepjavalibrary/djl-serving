# DJL Serving 控制台

DJL Serving 控制台是一个能实现模型管理，日志管理，依赖管理，配置管理等功能的DJL模型服务器管理平台，为用户使用DJL模型服务器最大程度的降低了操作难度，我们将提供包括下列基本功能:

* 模型管理
  * 模型列表
  * 模型注册
  * 模型卸载
  * 模型修改
  * 模型推理
* 日志管理
  * 日志列表
  * 日志详情
  * 日志下载  
* 依赖管理
  * 依赖列表
  * 依赖添加
  * 依赖删除  
* 配置管理
* 重启服务

## 模型管理
用户可以通过友好的界面来注册模型，查看系统所有模型的运行状态以及每个模型的详情，还可以在详情页面对模型进行`bacth_size`调整、设备扩容等高级配置操作。在模型推理界面提供文件上传、文本输入、自定义Headers等功能来编辑输入数据，对于模型输出数据，推理界面也能直接展示图片、json文本等常规数据以及下载其他流数据文件。

### 模型列表
模型列表展示了模型名称、模型版本号、模型状态以及模型的一些常用操作按钮（推理、卸载、修改），方便用户以统一的视图管理所有的模型。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/mode-list.png)

### 模型注册
可以通过主页**Add Model**按钮进入模型注册页面，只需要指定模型`url`、`model_name`、`version`、`engine`就可以快速注册模型。控制台提供了文本输入和文件上传两种方式来实现模型`url`的指定，可以输入线上的模型地址，也可以输入本地的模型地址，还可以将本地打包好的模型zip文件上传。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/add-model.png)

在高级设置区域，可以对模型的`batch_size`、`min_worker`、`max_worker`进行设置

### 模型卸载
当注册的模型出现错误需要删除时，可以通过点击模型列表中的删除按钮![img.png](https://resources.djl.ai/images/djl-serving/management_console/delete-btn.png)，确认后即可卸载注册失败的模型。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/delete-model.png)

### 模型修改
模型注册成功后，如果需要对模型的参数进行调整，可以通过模型列表中模型区域操作栏的![img.png](https://resources.djl.ai/images/djl-serving/management_console/update-btn.png)按钮进入模型修改页面，对模型进行`batch_size`、`min_worker`、`max_worker`以及`Device`进行调整。在Worker Groups界面还可以查看每个worker的健康状态和设备类型。 

![img.png](https://resources.djl.ai/images/djl-serving/management_console/update-model.png)

### 模型推理
![img.png](https://resources.djl.ai/images/djl-serving/management_console/inference-btn.png)推理按钮位于模型列表中模型区域下操作栏左侧，在模型推理界面，分为输入数据，输入结果两部分。

输入数据分为**Headers**、**Body**两部分，可以根据自己的需要自定义报文的header,如Accept、Content-Type；

![img.png](https://resources.djl.ai/images/djl-serving/management_console/header.png)

**Body**部分，控制台提供了文件上传、文本输入两种方式来编辑报文体数据，文件上传模式还提供了自定义Form表单参数的文本框，可以根据需要自定义出文件数据外的其他表单数据。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/body.png)

**Result**区域主要用于展示模型输出数据或者错误提示，如输出数据Content-type为**image/**，页面将直接展示图片内容；如果如输出数据Content-type为**application/json**，页面会直接呈现格式化好的json文本；如果是其他一些文件流数据，页面将会下载文件。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/result.png)

## 日志管理
用户可以直接在控制台上查看系统实时日志，不用再到服务器上通过命令行查看日志，大大减轻了查日志的工作量。

### 日志列表
通过主页导航栏**Log**进入，列表将展示系统日志、请求日志、模型日志等实时日志。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/log-list.png)

### 日志详情
通过日志列表的**Details**按钮进入日志详情页面，控制台将展示最近200行日志数据。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/log-detail.png)

### 日志下载
除了页面查看日志，控制台还提供了日志下载功能，让用户能查看当天所有的日志。

## 依赖管理
控制台为用户提供一个管理界面查看添加的第三方依赖jar包或者引擎相关依赖，用户还可以通过这个管理界面添加依赖或者删除依赖

### 依赖列表
通过主页导航栏**System**下二级菜单**Dependency**进入到列表界面，界面展示已经添加依赖jar文件

![img.png](https://resources.djl.ai/images/djl-serving/management_console/dependency-list.png)

如图所示，当前server已加载了pytorch、tensorflow、mxnet等引擎的相关依赖。

### 依赖添加

入口为依赖列表的**Add Dependency**按钮，控制台提供了**Engine**和**Jar**两种添加方式，
如需添加相关引擎的依赖，只需选择对应的引擎名即可下载相关依赖并加载到server。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/engine.png)

如果想添加公共库上的一些第三方jar包，可以在**Add form**选择**Maven**，填写相关的Maven坐标信息，server就会自动到Maven公共仓库下载jar包并加载

![img.png](https://resources.djl.ai/images/djl-serving/management_console/maven.png)

除了以上两种方式，用户还可以上传自己的jar包到server

![img.png](https://resources.djl.ai/images/djl-serving/management_console/upload-jar.png)

### 依赖删除
在依赖列表界面，每条依赖数据都会有一个**Delete**按钮，点击后server删除指定的jar文件，此时server并未卸载相关依赖，需要**重启server**才生效

## 配置管理
控制台提供了一个vscode风格的在线编辑器，对config.properties进行编辑，可以修改server的`max_request_size`、请求跨域等配置，修改保存后需**重启server**配置才会生效。

![img.png](https://resources.djl.ai/images/djl-serving/management_console/config.png)

## 重启服务
用户修改配置后需要重启server才会生效，控制台同样提供重启server的功能。
点击主页右上角的下拉菜单**Restart**，控制台会给server发送重启指令重启服务，仅需不到10秒的时间就可以完成server重启，重启功能仅支持**Docker环境**

![img.png](https://resources.djl.ai/images/djl-serving/management_console/restart.png)
