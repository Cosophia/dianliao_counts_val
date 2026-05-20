from pathlib import Path
from code_ import utils
from datastruct import Analyse
# command for install package
# C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple


if __name__ == '__main__':
    # param2: date: (1) 数字：最后的几个日期 (2) 单个日期：从该日期往后的所有 (3)两个日期：日期区间
    # param3: 模型个数
    # param4: 数据量限制: (1)整数：只跑这个数量的图 (2)0-1的小数：百分比
    date_mode = input(
          "(模式1) 数字：最后的几个日期 \n"
          "(模式2) [日期,)：从该日期往后的所有 \n"
          "(模式3) [日期1,日期2]：日期区间\n"
          "( 0 ) 退出当前程序！\n"
          "请选择日期模式 (输入对应序号即可)："
          )
    if date_mode == "0" :
        print("当前程序已退出！")
        exit()
    elif date_mode == "1" :
        date_num = int(input(f"您选择的的是模式{date_mode}：最后的X个日期，请输入您期望的最后几个日期:"))
        model_num = int(input("请选择模型个数："))
        Analyse(
            folder = Path(utils.FOLDER),
            model_folder = Path(utils.MODEL_FOLDER),
            date = date_num,
            latest_model = model_num,
            constraint=0,)
    elif date_mode == "2" :
        actual_date = str(input(f"您选择的的是模式{date_mode}：从该日期往后的所有,请输入某个具体的日期(如260429):"))
        model_num = int(input("请选择模型个数："))
        Analyse(
            folder=Path(utils.FOLDER),
            model_folder = Path(utils.MODEL_FOLDER),
            date = actual_date,
            latest_model = model_num,
            constraint=0, )
    elif date_mode == "3" :
        print(f"您选择的的是模式{date_mode}：日期区间")
        start_date = str(input("请输入起始日期："))
        end_date = str(input("请输入截止日期："))
        model_num = int(input("请选择模型个数："))
        Analyse(
            folder=Path(utils.FOLDER),
            model_folder = Path(utils.MODEL_FOLDER),
            date = (start_date,end_date),
            latest_model = model_num,
            constraint=0, )
        # 若有其他匹配模式自行扩张
        # elif  date_mode == "4" :
    else:
        print("输入有误！")
        

    # def pickle_ver_change(path: Path):
    #     with open(path, "rb") as f:
    #         d = pickle.load(f)
    #
    #     for k in d:
    #         d[k] = len(d[k][0])
    #
    #     with open(path, "wb") as f:
    #         pickle.dump(d, f)
    #
    # for path in Path(r"D:\counting_analysis\result").rglob('record.pkl'):
    #     pickle_ver_change(path)
    #     print("ok")
