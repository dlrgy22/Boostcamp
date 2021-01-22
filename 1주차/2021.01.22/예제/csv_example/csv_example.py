import csv
USA_data = []
rownum = 0

with open("customers.csv", "r", encoding="utf8") as c_file:
    csv_data = csv.reader(c_file)
    for row in csv_data:
        if rownum == 0:
            header = row
        location = row[10]
        if location == "USA":
            USA_data.append(row)
        rownum += 1

with open("USA_customer.csv", "w", encoding="utf8") as u_c_file:
    writer = csv.writer(u_c_file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    for data_element in USA_data:
        writer.writerow(data_element)

