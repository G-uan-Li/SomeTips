import requests
import pandas as pd

data = {
    'ISIN': [],
    'Bond Code': [],
    'Issuer': [],
    'Bond Type': [],
    'Issue Date': [],
    'Latest Rating': [],
}

url = 'https://iftp.chinamoney.com.cn/ags/ms/cm-u-bond-md/BondMarketInfoListEN'
headers = {
    'Cookie': 'AlteonP10=AC3rTyw/F6zcwvFeUcQVOg$$; apache=4a63b086221745dd13be58c2f7de0338; ags=2a1ba4d47b619c011c19c1cc4b3c0c32; _ulta_id.ECM-Prod.ccc4=454356464cd2a31d; _ulta_ses.ECM-Prod.ccc4=c9ec1271ca4edbab',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
}

for i in range(1, 9):
    datas = {
        'pageNo': f'{i}',
        'pageSize': '15',
        'isin': '',
        'bondCode': '',
        'issueEnty': '',
        'bondType': '100001',
        'couponType': '',
        'issueYear': '2023',
        'rtngShrt': '',
        'bondSpclPrjctVrty': '',
    }
    r = requests.post(url=url, headers=headers, data=datas).json()

    resultList = r['data']['resultList']
    for result in resultList:
        isin = result['isin']
        bondCode = result['bondCode']
        entyFullName = result['entyFullName']
        bondType = result['bondType']
        issueStartDate = result['issueStartDate']
        debtRtng = result['debtRtng']

        data['ISIN'].append(isin)
        data['Bond Code'].append(bondCode)
        data['Issuer'].append(entyFullName)
        data['Bond Type'].append(bondType)
        data['Issue Date'].append(issueStartDate)
        data['Latest Rating'].append(debtRtng)

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False, encoding='utf-8-sig')
