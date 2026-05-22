# dianliao_counts_val
Here is the code of  Validation of the counting dianliao! Welcome to use and mention the faults U see in the code.

### Version1.0:对代码尽可能解耦。
* 将Analyse.py 放datastruct中，后续随着对代码的深入与了解，会将项目的结构进行重构使其更加工程化。
### Version2.0:修改代码的一些泛化问题：
* ① cv2.imread()无法读取中文路径，利用cv2.imdecode()解决。
* ② 修改了主函数的日期选择，使得进程可以选择期望的日期模型过滤日期。
* ③ 添加了模型路径变量于utils.py中，方便使用者修改。
### Version2.1 :修改了读取图片匹配文件名
* 通过img.name.count("_")获取文件名的"\_"个数，读取不同有规律的文件名。