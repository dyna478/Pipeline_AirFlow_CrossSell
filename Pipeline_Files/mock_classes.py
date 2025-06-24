"""
Mock classes for Airflow dependencies
"""
import logging

logger = logging.getLogger("mock_airflow")

class Variable:
    """Mock for Airflow Variable class"""
    _vars = {
        'TELUS_API_KEY': 'test_api_key',
        'TELUS_USE_TEST': 'true'
    }
    
    @classmethod
    def get(cls, key, default=''):
        """Get a variable value"""
        logger.info(f"Variable.get({key}, {default})")
        return cls._vars.get(key, default)
    
    @classmethod
    def set(cls, key, value):
        """Set a variable value"""
        logger.info(f"Variable.set({key}, {value})")
        cls._vars[key] = value

class XCom:
    """Mock for Airflow XCom"""
    _data = {}
    
    @classmethod
    def push(cls, key, value, task_id=None, dag_id=None, execution_date=None):
        """Push a value to XCom"""
        if task_id not in cls._data:
            cls._data[task_id] = {}
        cls._data[task_id][key] = value
        return True
    
    @classmethod
    def pull(cls, task_id=None, key=None, dag_id=None, execution_date=None):
        """Pull a value from XCom"""
        if task_id in cls._data and key in cls._data.get(task_id, {}):
            return cls._data[task_id].get(key)
        return None

class TaskInstance:
    """Mock for Airflow TaskInstance"""
    def __init__(self, task_id='mock_task'):
        self.task_id = task_id
        self._xcom_data = {}
    
    def xcom_push(self, key, value):
        """Push a value to XCom"""
        self._xcom_data[key] = value
        logger.info(f"TaskInstance({self.task_id}).xcom_push({key})")
        return True
    
    def xcom_pull(self, task_ids=None, key=None):
        """Pull a value from XCom"""
        logger.info(f"TaskInstance({self.task_id}).xcom_pull(task_ids={task_ids}, key={key})")
        return self._xcom_data.get(key)

class DAGContext:
    """Mock for Airflow DAG run context"""
    def __init__(self, execution_date=None, dag_id='mock_dag'):
        from datetime import datetime
        self.execution_date = execution_date or datetime.now()
        self.dag_id = dag_id
        self.ds = self.execution_date.strftime('%Y-%m-%d')
        self.ts = self.execution_date.strftime('%Y-%m-%dT%H:%M:%S')
        self.task_instance = TaskInstance()
        self.dag_run = type('obj', (object,), {
            'run_id': f"mock_run_{self.ds}",
            'execution_date': self.execution_date
        })
    
    def get_context(self):
        """Get the context dict"""
        return {
            'execution_date': self.execution_date,
            'ds': self.ds,
            'ts': self.ts,
            'dag_id': self.dag_id,
            'task_instance': self.task_instance,
            'dag_run': self.dag_run
        } 

model_path = './models/model.pkl'  # Update this path 