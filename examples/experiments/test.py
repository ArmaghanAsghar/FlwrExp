import time



start_time = time.time()
time.sleep(2)
end_time = time.time()

elapsed_time = end_time - start_time

print("Aggregation Time: " f'{elapsed_time*1000} ms')