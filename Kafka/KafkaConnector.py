from kafka import KafkaConsumer, TopicPartition
from time import sleep
import pandas as pd
import json

class KafkaConnector:
    def __init__(self, KAFKA_BROKER_URL = 'localhost:9092', groud_id='0'):
        self.KAFKA_BROKER_URL = KAFKA_BROKER_URL
        self.groud_id = groud_id


    def load_data(self):
        self.consumer = KafkaConsumer(group_id=self.groud_id,
                                        bootstrap_servers=self.KAFKA_BROKER_URL,
                                        auto_offset_reset='earliest',
                                        value_deserializer=lambda v: json.loads(v.decode('utf-8')))

        last_offset = self.consumer.end_offsets([TopicPartition('test', 0)])
        last_offset = list(last_offset.values())[0]

        self.consumer.assign([TopicPartition('test', 0)])
        if self.consumer.position(TopicPartition('test', 0)) == last_offset:
            print('It`s empty')
            self.consumer.close()

            return pd.DataFrame([])

        data = pd.DataFrame([])

        for msg in self.consumer:
            print (msg.value)
            message_df = pd.DataFrame([msg.value])
            data = pd.concat([data, message_df], axis=0, ignore_index=True)

            self.consumer.commit()
            last_offset = self.consumer.end_offsets([TopicPartition('test', 0)])
            last_offset = list(last_offset.values())[0]
            print(last_offset)
            if msg.offset == last_offset-1:

                #do preprocessing

                print('I am vse prochital!')
                print(data)
                break

        self.consumer.close()

        return data

if __name__=='__main__':
    KAFKA_BROKER_URL = '192.168.1.138:9092'
    group_id='0'

    kafka_con = KafkaConnector()
    kafka_con.load_data()

####
"""
KAFKA_BROKER_URL = '192.168.1.138:9092'


consumer = KafkaConsumer(group_id='0', bootstrap_servers=KAFKA_BROKER_URL, auto_offset_reset='earliest')

last_offset = consumer.end_offsets([TopicPartition('test', 0)])
last_offset = list(last_offset.values())[0]
print(last_offset)

consumer.assign([TopicPartition('test', 0)])
if consumer.position(TopicPartition('test', 0)) == last_offset:
    print('It`s empty')
    consumer.close()
    exit()

for msg in consumer:
    print (msg.value)
    consumer.commit()
    last_offset = consumer.end_offsets([TopicPartition('test', 0)])
    last_offset = list(last_offset.values())[0]
    print(last_offset)
    if msg.offset == last_offset-1:

        #do preprocessing

        print('I am vse prochital!')
        break

consumer.close()
"""
