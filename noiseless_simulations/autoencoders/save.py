name1 = "res/gyro_output.m"
name2 = "res/gyro_real_output.m"

with open(name1, "w+") as f:
    f.write("y = [" + str(logY[0]) + ",\n")
    for i in range(1, logY.size - 1):
        f.write(str(logY[i]) + ",\n")
    f.write(str(logY[logY.size - 1]) + "];\n")

with open(name2, "w+") as f:
    f.write("yr = [" + str(logYR[0]) + ",\n")
    for i in range(1, logY.size - 1):
        f.write(str(logYR[i]) + "\n")
    f.write(str(logYR[logY.size - 1]) + "];\n")
