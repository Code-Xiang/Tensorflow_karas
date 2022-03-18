import requests
import bs4

def open_url(url):
    #使用代理 proxies(计算机科学技术 代理服务)
    # proxies={"http":"127.0.1:1080","https":"127.0.0.1:1080"}
    headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39'}
    #res = requests.get(url,headers=hearders,proxies=proxies)
    res=requests.get(url,headers=headers)

    return res
def find_movies(res):
    soup=bs4.BeautifulSoup(res.text,'html.parser')
    #电影名
    movies=[]
    targets=soup.find_all("div",class_="hd")
    for each in targets:
        movies.append(each.a.span.text)
    
    #评分
    ranks=[]
    targets=soup.find_all("sapn",class_="rating_num")
    print(targets)
    for each in targets:
        ranks.append('评分：%s'%each.text)
    
    #资料
    messages = []
    targets = soup.find_all("div", class_="bd")
    for each in targets:
        try:
            messages.append(each.p.text.split(
                '\n')[1].strip()+each.p.text.split('\n')[2].strip())
        except:
            continue

    result=[]
    length=len(movies)

    for i in range(length):
        result.append(movies[i]+'\n' + messages[i]+'\n' )
        
    return result

#找出一共有多少个页面
def find_depth(res):
    soup=bs4.BeautifulSoup(res.text,'html.parser')
    depth=soup.find('span',class_='next').previous_sibling.previous_sibling.text

    return int(depth)

def main():
    host="https://movie.douban.com/top250"
    res = open_url(host)
    depth=find_depth(res)

    result=[]
    for i in range(depth):
        url = host+'/?start='+str(25*i)
        res = open_url(url)
        result.extend(find_movies(res))

    with open("豆瓣top250电影.txt","w",encoding="utf-8") as file:
       for each in result:
           file.write(each) 

if __name__=="__main__":
    main()


