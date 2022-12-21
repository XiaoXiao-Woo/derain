from configs.configs import TaskDispatcher
from UDL.AutoDL.trainer import main
from common.builder import build_model, derainSession

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='derain', mode='entrypoint', arch='DFTLW')
    print(TaskDispatcher._task.keys())
    main(cfg, build_model, derainSession)