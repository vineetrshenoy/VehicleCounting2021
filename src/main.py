import configparser
import app_logger

logger = app_logger.get_logger('main')

config = configparser.ConfigParser()
config.read('config/basic.ini')



def main() -> None:
    
    logger.info('Setting environment variable')
    print(config['DEFAULT']['TEST'])



if __name__ == '__main__':

    main()