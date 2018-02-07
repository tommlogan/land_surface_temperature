import logging
import logging.config

config = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'code/logging.log',
            'mode': 'w',
            'formatter': 'detailed',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console','file']
    },
}

logging.config.dictConfig(config)
