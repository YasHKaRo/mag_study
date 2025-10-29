# Networks
Более подробнее про [[Linux]] сети от ПСБ
### Глоссарий

**TCP/IP** — сетевая модель передачи данных, представленных в цифровом виде. Модель описывает способ передачи данных от источника информации к получателю. В модели предполагается прохождение информации через четыре уровня, каждый из которых описывается правилом (протоколом передачи). 

**VPN** (virtual private network) — обобщённое название технологий, позволяющих обеспечить одно или несколько сетевых соединений поверх чьей-либо другой сети. 

**BGP** (Border Gateway Protocol, протокол граничного шлюза) — протокол динамической маршрутизации. 

**OSPF** (Open Shortest Path First) — протокол динамической маршрутизации, основанный на технологии отслеживания состояния канала (link-state technology) и использующий для нахождения кратчайшего пути алгоритм Дейкстры. 

**IP-адрес** (Internet Protocol) — уникальный числовой идентификатор устройства в компьютерной сети, работающей по протоколу IP. 

**MAC-адрес** (Media Access Control — надзор за доступом к среде, также Hardware Address, также физический адрес) — уникальный идентификатор, присваиваемый каждой единице сетевого оборудования или некоторым их интерфейсам в компьютерных сетях Ethernet. 

**ICMP** (Internet Control Message Protocol — протокол межсетевых управляющих сообщений) — сетевой протокол, входящий в стек протоколов TCP/IP. В основном ICMP используется для передачи сообщений об ошибках и других исключительных ситуаций, возникших при передаче данных, например, запрашиваемая услуга недоступна или хост или маршрутизатор не отвечают. 

**ARP** (Address Resolution Protocol — протокол определения адреса) — протокол в компьютерных сетях, предназначенный для определения MAC-адреса другого компьютера по известному IP-адресу. 

**DNS** (Domain Name System «система доменных имён») — компьютерная распределённая система для получения информации о доменах. Чаще всего используется для получения IP-адреса по имени хоста (компьютера или устройства), получения информации о маршрутизации почты и/или обслуживающих узлах для протоколов в домене (SRV-запись). 

**cURL** (client URL или Curl URL Request Library)— кроссплатформенная служебная программа командной строки, позволяющая взаимодействовать с множеством различных серверов по множеству различных протоколов с синтаксисом URL. 

**TELNET** (Teletype Network) — сетевой протокол для реализации текстового терминального интерфейса по сети (в современной форме — при помощи транспорта TCP). Название «telnet» имеют также некоторые утилиты, реализующие клиентскую часть протокола.

**HTTP** (HyperText Transfer Protocol) — сетевой протокол прикладного уровня, который изначально предназначался для получения с серверов гипертекстовых документов в формате HTML, а с течением времени стал универсальным средством взаимодействия между узлами как Всемирной паутины, так и изолированных веб-инфраструктур.


Урок предполагает знакомство со стеком TCP/IP v4. В открытом доступе имеются хорошие ресурсы для его изучения, например:
* Курс МФТИ по компьютерным сетям - https://www.youtube.com/playlist?list=PLthfp5exSWErPFK_-EAhVtxO3XoY0gsSe

## Сетевые интерфейсы
Сетевое взаимодействие в Линукс осуществляется посредством сетевых интерфейсов. Сетевые интерфейсы могут быть как физическими (ethernet / wifi сетевые карты), так и виртуальнымии (интерфейсы VPN, docker). Интерфейсу может быть присвоен один или несколько IP адресов с указанием маски подсети. Физические интерфейсы автоматически определяются системой при старте, виртуальные создаются запущенными процессами или самой системой.

Маршрутизация трафика между интерфейсами осуществляется при помощи таблиц(ы) маршрутизации, опций Линукса (такой как net.ipv4.ip_forward) и правил iptables, nftables.

Рассмотрим сетевые интерфейсы типичной виртуальной машины (st{номер})
```shell
ip address
---
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether d0:0d:17:a8:55:8c brd ff:ff:ff:ff:ff:ff
    altname enp138s0
    altname ens8
    inet 10.1.0.27/16 metric 100 brd 10.1.255.255 scope global dynamic eth0
       valid_lft 4294966459sec preferred_lft 4294966459sec
    inet6 fe80::d20d:17ff:fea8:558c/64 scope link 
       valid_lft forever preferred_lft forever


```
Интерфейс lo (loopback) - автоматически созданный виртуальный интерфейс, его основное назначение в серьезных компьютерных сетях - наличие всегда включенного интерфейса, на который можно повесить IP адрес, использующийся в протоколах маршрутизации (BGP, OSPF).  В обычных задачах такая функция не востребована и он используется в основном для работы с сетевыми сервисами "внутри" самой машины (например, для запуска веб сервера, доступ к которому есть только с самой машины и отсутствует извне).

Интерфейс eth0 - на данной ВМ используется для выхода в "большой" интернет. Адрес 10.1.0.27/16 (/16 - маска подести) означает, что IP адрес самого интерфейса 10.1.0.27, а через данный интерфейс можно обращаться к ip адресам из диапазона  10.1.0.1 - 10.1.255.254.

## Таблица маршрутизации
Посмотрим на таблицу маршрутизации
```shell
ip route
---
default via 10.1.0.1 dev eth0 
10.1.0.0/16 dev eth0 proto kernel scope link src 10.1.0.27 metric 100 
10.1.0.1 dev eth0 proto dhcp scope link src 10.1.0.27 metric 100 
10.1.0.2 dev eth0 proto dhcp scope link src 10.1.0.27 metric 100 


```
Здесь определен дефолтный мартшрут через 10.1.0.1 (этот IP адрес доступен через интерфейс eth0) и специфические маршруты для отдельных подсетей (10.1.0.0/16). Дефолтный маршрут выбирается, если IP адрес назначения не находится в диапазоне более специфических маршрутов. Таким образом, запросы к внешним интернет ресурсам будут направлены через интерфейс eth0 на адрес 10.1.0.1, ip пакеты с адресом назначения в диапазоне 10.1.0.0/16 также пойдут через интерфейс eth0.

## Виртуальные интерфейсы
Список сетевых интерфейсов ВМ это не что-то неизменное. Разные приложения могут легко добавлять свои интерфейсы. Например, при установке docker добавляет свой интерфейс для связи с контейнерами.

Установим пакет docker.io и посмотрим, как это повлияет на наше сетевое окружение
```shell
sudo apt install docker.io
```
Добавился интерфейс docker0, через который направляется трафик из подсети 172.17.0.0/16 (в контейнеры docker), а в таблицу маршрутизации добавилось правило маршрутизации трафика в подсеть 172.17.0.0/16
```shell
ip a
---
...
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:33:96:81:79 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
       
ip route
---
...
172.17.0.0/16 dev docker0 proto kernel scope link src 172.17.0.1 linkdown
```
## Проверка сетевой связности
Проверить сетевую связность до IP адреса можно с помощью команды ping (нужно принять во внимание, что иногда на хостах блокируются icmp ответы, так что отсутствие ответа еще не означает недоступность хоста)
```shell
ping 1.1.1.1 -c 4
---
PING 1.1.1.1 (1.1.1.1) 56(84) bytes of data.
64 bytes from 1.1.1.1: icmp_seq=1 ttl=56 time=8.07 ms
64 bytes from 1.1.1.1: icmp_seq=2 ttl=56 time=7.85 ms
64 bytes from 1.1.1.1: icmp_seq=3 ttl=56 time=8.80 ms
64 bytes from 1.1.1.1: icmp_seq=4 ttl=56 time=8.78 ms

--- 1.1.1.1 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3007ms
rtt min/avg/max/mdev = 7.848/8.373/8.798/0.420 ms
```
Если icmp ответы на хосте заблокированы, то сетевая связность до хоста может быть проверена, если он находится в одной из подсетей сетевых интерфейсов машины, при помощи команды arp. Ведь для поиска MAC адреса хоста наша машина будет рассылать arp запросы ("у кого такой-то ip адрес?"), результаты которых поместит в arp таблицу.

Установим нужный пакет для доступа к команде arp
```shell
sudo apt install net-tools
```
Проверим записи в arp таблице
```shell
arp -an
---
? (10.1.0.9) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.1) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.2) at 00:00:5e:00:01:00 [ether] on eth0
```
Теперь пропингуем машину с gitlab и nexus и снова посмотрим на вывод команды `arp -an`
```shell
ping gitlab -c 1
---
PING gitlab.ru-central1.internal (10.1.0.15) 56(84) bytes of data.
64 bytes from gitlab.ru-central1.internal (10.1.0.15): icmp_seq=1 ttl=61 time=24.4 ms


ping nexus -c 1
---
PING nexus.ru-central1.internal (10.1.0.31) 56(84) bytes of data.
64 bytes from nexus.ru-central1.internal (10.1.0.31): icmp_seq=1 ttl=61 time=1.10 ms


arp -an
---
? (10.1.0.9) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.1) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.2) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.15) at 00:00:5e:00:01:00 [ether] on eth0
? (10.1.0.31) at 00:00:5e:00:01:00 [ether] on eth0

```
В arp таблицу добавилось два адреса (10.1.0.15 и 10.1.0.31). Как уже упоминалось, они добавляются в arp таблицу, даже если бы icmp ответы на хостах gitlab и nexus были заблокированы.

Более "современным" аналогом команды `arp` является `ip neigh`
```shell
ip neigh
10.1.0.9 dev eth0 lladdr 00:00:5e:00:01:00 STALE
10.1.0.1 dev eth0 lladdr 00:00:5e:00:01:00 REACHABLE
10.1.0.2 dev eth0 lladdr 00:00:5e:00:01:00 STALE
10.1.0.15 dev eth0 lladdr 00:00:5e:00:01:00 STALE
10.1.0.31 dev eth0 lladdr 00:00:5e:00:01:00 STALE
```


Часто недоступность IP адреса вызвана отсутствием нужного маршрута. Проверим, как отразится удаление маршрута по умолчанию на доступности внешних ресурсов. Вначале добавим маршрут в сеть 192.168.1.0/24 (связана с wireguard), чтобы не потерять доступ к нашей машине после удаления дефолтного
```shell
sudo ip route add 192.168.1.0/24 dev eth0
```

А теперь удалим дефолтный маршрут 
```shell
sudo ip route del default via 10.1.0.1 dev eth0
```
После удаления дефолтного маршрута таблица маршрутизации выглядит так:
```shell
ip route
---
10.1.0.0/16 dev eth0 proto kernel scope link src 10.1.0.27 metric 100 
10.1.0.1 dev eth0 proto dhcp scope link src 10.1.0.27 metric 100 
10.1.0.2 dev eth0 proto dhcp scope link src 10.1.0.27 metric 100 
172.17.0.0/16 dev docker0 proto kernel scope link src 172.17.0.1 linkdown 
192.168.1.0/24 dev eth0 scope link 
```
С такой таблицей маршрутизации у Линукса нет информации, куда отправлять пакеты с адресом назначения не из сетей 10.1.0.0/16, 192.168.1.0/24 и 172.17.0.0/16, а значит они просто будут отброшены.

Повторим команду ping снова - icmp запрос не отправляется, сетевой связности до 1.1.1.1 нет из-за отсутствия маршрута по умолчанию
```shell
ping 1.1.1.1 -c 4      
---                               
ping: connect: Network is unreachable
```
Восстановим удаленный маршрут для восстановления сетевой связности
```shell
sudo ip route add default via 10.1.0.1 dev eth0
```
Теперь пинг опять работает
```shell
ping 1.1.1.1 -c 2
---
PING 1.1.1.1 (1.1.1.1) 56(84) bytes of data.
64 bytes from 1.1.1.1: icmp_seq=1 ttl=54 time=4.94 ms
64 bytes from 1.1.1.1: icmp_seq=2 ttl=54 time=4.06 ms
```
## Сетевые соединения
### Проверка доступности сервиса
Для установления сетевого соединения с конкретным приложением помимо IP адреса используется понятие порта (число в диапазоне 1 - 65535). Установленное соединение характеризуется типом (tcp, udp), IP адресом клиента, IP адресом сервера, портом клиента (назначается системой произвольно), портом сервера (задается сервером).

Установленные соединения можно посмотреть командами netstat или ss (более современная версия netstat). Откроем еще одно ssh соединение до нашей ВМ

В одной консоли установим соединение до сервера gitlab
```shell
telnet gitlab 443
```
В другой консоли поищем установленное соединение
```shell
ss -nt | grep 443
--- 
ESTAB 0      0          10.1.0.27:51556     10.1.0.15:443  
```
Видим установленное соединение (tcp, потому что в команде был указан флаг -t), отличающиеся портом клиента. ESTAB == ESTABLISHED.

Таким образом, если до сервиса хоста имеется установленное соединение (статус ESTABLISHED), то сервис доступен по сети.

telnet можно использовать для проверки доступности любого сервиса (здесь проверяется доступность ssh к rds на 22 порту)
```shell
telnet rds 22
---
telnet rds 22
Trying 10.132.0.5...
Connected to rds.ru-central1.internal.
Escape character is '^]'.
SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13.11
```
видно, что соединение установилось, что также подтверждается командой ss, запущенной в другом окне терминала (новое соединение с клиентским портом 54566). Два других соединения - это ssh соединения от клиента к ВМ через туннель wireguard
```shell
ss -nt | grep 22
---
ESTAB 0      0          10.1.0.27:22    192.168.1.100:44040       
ESTAB 0      0          10.1.0.27:54566    10.132.0.5:22          
ESTAB 0      52         10.1.0.27:22    192.168.1.100:41348   

```
Интересно, что используя лишь telnet можно осуществить http подключение:

```shell
telnet ya.ru 80
---
Trying 5.255.255.242...
Connected to ya.ru.
Escape character is '^]'.

# Вводим следующий текст и после строки Host два раза нажимаем на Enter
GET / HTTP/1.1
Host: ya.ru

# Ответ сервера
HTTP/1.1 301 Moved permanently
Transfer-Encoding: chunked
P3P: policyref="/w3c/p3p.xml", CP="NON DSP ADM DEV PSD IVDo OUR IND STP PHY PRE NAV UNI"
X-Content-Type-Options: nosniff
NEL: {"report_to": "network-errors", "max_age": 100, "success_fraction": 0.001, "failure_fraction": 0.1}
Report-To: { "group": "network-errors", "max_age": 100, "endpoints": [{"url": "https://dr.yandex.net/nel", "priority": 1}, {"url": "https://dr2.yandex.net/nel", "priority": 2}]}
Cache-Control: max-age=86400,private
set-cookie: is_gdpr=0; Path=/; Domain=.ya.ru; Expires=Sun, 20 Jun 2027 14:14:06 GMT
set-cookie: is_gdpr_b=CNmGWBCpxwI=; Path=/; Domain=.ya.ru; Expires=Sun, 20 Jun 2027 14:14:06 GMT
set-cookie: _yasc=mJ5ByZjM9OygVZPW8qI2Cq/4/3Nf0zjo6RB7Gntun3vUdggNue6ABPzmhP638azLqrQ=; domain=.ya.ru; path=/; expires=Mon, 18 Jun 2035 14:14:06 GMT; secure
Accept-CH: Sec-CH-UA-Platform-Version, Sec-CH-UA-Mobile, Sec-CH-UA-Model, Sec-CH-UA, Sec-CH-UA-Full-Version-List, Sec-CH-UA-WoW64, Sec-CH-UA-Arch, Sec-CH-UA-Bitness, Sec-CH-UA-Platform, Sec-CH-UA-Full-Version, Sec-CH-Viewport-Width, Viewport-Width, DPR, Device-Memory, RTT, Downlink, ECT, Width
X-Yandex-Req-Id: 1750428846127392-17832575287833089215-balancer-l7leveler-kubr-yp-sas-5-BAL
Portal: Home
Date: Fri, 20 Jun 2025 14:14:06 GMT
Location: https://ya.ru/

```
### Проверка портов, открытых сервисом
Посмотрим как выглядят порты, открываемые сервисами, запущенными на машине. Запустим сервис на порту 3000 на всех интерфейсах (т.е. к сервису можно обратиться с любого интерфейса - как извне, так и изнутри машины)
```shell
nc -l 3000
```
в другом терминале на той же машине выполним команду
```shell
ss -tnlp | grep 3000
---
LISTEN    0         1                  0.0.0.0:3000             0.0.0.0:*        users:(("nc",pid=1184,fd=3))
```
`0.0.0.0:3000` означает, что процесс `nc` слушает 3000 порт на всех интерфейсах, в том числе к нему можно обратиться извне (например, с rds) командой nc _ip адрес сервиса_ 3000, либо внутри машины командой `nc localhost 3000` (из другого терминала). 

Теперь, если начать набирать текст в терминале с запущенной командой  `nc localhost 3000`, этот текст начнет появляться в терминале с командой `nc -l 3000`.

Можно запустить сервер с привязкой только к конкретному интерфейсу
```shell
nc -l 127.0.0.1 3000
```
тогда вывод `ss -tnlp | grep 3000` изменится
```shell
ss -tnlp | grep 3000
---
LISTEN    0         1                127.0.0.1:3000             0.0.0.0:*        users:(("nc",pid=99935,fd=3))
```
видно, что теперь сервер слушает только на localhost, обращение к нему изнутри машины доступно, а извне нет.

## Проверка работы службы DNS
Работу службы DNS (преобразование доменного имени в IP адрес) на машине можно проверить той же командой `ping`
```shell
ping ya.ru -c 2
---
PING ya.ru (5.255.255.242) 56(84) bytes of data.
64 bytes from ya.ru (5.255.255.242): icmp_seq=1 ttl=245 time=8.50 ms
64 bytes from ya.ru (5.255.255.242): icmp_seq=2 ttl=245 time=8.64 ms

--- ya.ru ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1002ms
rtt min/avg/max/mdev = 8.497/8.567/8.638/0.070 ms
```
видно, что `ping` нашла IP адрес 5.255.255.242,

командой `nslookup` (используется дефолтный сервер DNS, заданный в `/etc/resolv.conf`)
```shell
nslookup ya.ru
Server:		127.0.0.53
Address:	127.0.0.53#53

Non-authoritative answer:
Name:	ya.ru
Address: 5.255.255.242
Name:	ya.ru
Address: 77.88.55.242
Name:	ya.ru
Address: 2a02:6b8::2:242
```
либо командами `dig` или `host`.

Работу конкретного сервера DNS можно проверить так
```shell
nslookup ya.ru 8.8.8.8
---
Server:		8.8.8.8
Address:	8.8.8.8#53

Non-authoritative answer:
Name:	ya.ru
Address: 5.255.255.242
Name:	ya.ru
Address: 77.88.55.242
Name:	ya.ru
Address: 2a02:6b8::2:242
```

## cURL
cURL - всеобъемлющий инструмент, который умеет работать с массой протоколов. Подробная информация по нему есть в прикрепленных ссылках, мы лишь рассмотрим ключевые опции для работы с http(s), а также посмотрим на то, как он заменяет собой telnet.

Получаем содержимое http странички (`-v` - выводит дополнительную информацию при выполнении запроса, `-k` игнорирует ошибки с сертификатами https)
```shell
curl -v https://gitlab.psbschool.tech
---
...
HTTP/2 302
<html><body>You are being <a href="https://gitlab.psbschool.tech/users/sign_in">redirected</a>.</body></html>
```
Для того, чтобы последовать за редиректом используем опцию `-L`. Теперь уже получаем содержимое страницы
```shell
curl -v  -L https://gitlab.psbschool.tech
```
Опция `-O` позволяет скачать файл 
```shell
curl -O https://nexus.psbschool.tech/repository/shared-files/test.txt
cat test.txt
```
Манипулируем адресом по которому производится обращение, в заголовках запроса и для целей проверки сертификатов используется домен gitlab.psbschool.tech, а обращение производится к заданному ip адресу (10.1.0.31 - IP адрес Nexus). В данном случае эти ничего не дает, но подобные манипуляции иногда требуются для отладки.
```shell
curl -v https://gitlab.psbschool.tech --resolve 'gitlab.psbschool.tech:443:10.1.0.31'
---
* Added gitlab.psbschool.tech:443:10.1.0.31 to DNS cache
* Hostname gitlab.psbschool.tech was found in DNS cache
*   Trying 10.1.0.31:443...

* TCP_NODELAY set
* Connected to gitlab.psbschool.tech (10.1.0.31) port 443 (#0)


> GET / HTTP/1.1
> Host: gitlab.psbschool.tech
> User-Agent: curl/7.68.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: nginx/1.18.0 (Ubuntu)
< Date: Sat, 21 Jun 2025 09:49:57 GMT
< Content-Type: text/html
< Content-Length: 10028
< Connection: keep-alive
< X-Content-Type-Options: nosniff
< X-Frame-Options: DENY
< X-XSS-Protection: 1; mode=block
< Last-Modified: Sat, 21 Jun 2025 09:49:57 GMT
< Pragma: no-cache
< Cache-Control: no-cache, no-store, max-age=0, must-revalidate, post-check=0, pre-check=0
< Expires: 0

```
Удобный вывод (заголовок ответа и сам ответ)
```shell
curl -D - -s https://nexus.psbschool.tech
```


### Материалы
Mastering the curl command line
* https://www.youtube.com/watch?v=V5vZWHP-RqU
* https://daniel.haxx.se/media/mastering%20the%20curl%20command%20line.pdf


## SSH. Создание ключа для доступа на сервер

На хосте rds cоздадим новую пару ключей типа rsa, которую будем использовать для доступа к своей виртуальной машине (при создании ключей желательно задать пароль на доступ). 
```shell
# rds
cd ~
ssh-keygen -f myid_rsa -t rsa
---
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in myid_rsa
Your public key has been saved in myid_rsa.pub
The key fingerprint is:
SHA256:daq4V5ag3ZQ1hD3Jwty/0H+QtU9C2llVkS85TxUMIxc someone@rds
The key's randomart image is:
+---[RSA 3072]----+
|         o B.E+oO|
|          = @ oo+|
|          .+.O *+|
|        ..ooo %.+|
|       oS+.. . @.|
|      ....=   . =|
|      . .o      .|
|       ..        |
|      ..         |
+----[SHA256]-----+

```
В папке пользователя появились два ключ `myid_rsa` и  `myid_rsa.pub`. `myid_rsa` - приватный ключ, `myid_rsa.pub` - публичный. Для того, чтобы войти на ВМ (st{номер}) с использованием приватного ключа `myid_rsa`, содержимое публичного ключа `myid_rsa.pub` нужно добавить (дописать в конец файла) в ~/.ssh/authorized_keys на ВМ (st{номер}). Для пользователя user, созданного на ВМ, дописываем в конец файла /home/user/.ssh/authorized_keys:

```shell
... (пример, используйте свой публичный ключ) ...
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCx6VvMiKIsMoM8bwQIJmCsZj2CRGeFzwtRsmq6Wi2+hqIGBrkr9H+dVMvQi7J7h5h03C6JnZ4mThZU2xzXhDrom/b1H9O/RwPTlcQJ1HiRyC9PSnFmlGEiPmp2HXYi200xs8Bx3/icQIBSk/mj2sXqReGql/YWBpE4mGC0nV5W7TePnwvhB1mnQmJbx78CVZ2aSqp3RosSOP3EbXm2o0v3gLZdb7Z3pwSRrO/clLcxqnAwDmJ5W6NPDf8RFy/d841ruNs+uhpYi2UO61LEqGlCDBWXrmq2tHdzLy9cba/O0cerMuygfj/SixL4wFYRpd/vRXawI7eNxxgKE1/ATY3t/+R/bqeS/BWXsIzQTtRRXmrab/h5Q9M/ibsz7bb76tJBpb+/v0BzO4DVwY4wNwJPWfERefq5BWkqkEv2L+Lsha0ct/gAdksBSzuhDHTh2RwoW0kGyYsBL5ptzwGw7Md291RYbQNBK0gwM9NzOzXWkvuZLCDHmIhG4h1M1zzyIoM= vasilievlv@rds.ru-central1.internal
```

Предупреждение: если удалить уже существующий ключ в /home/user/.ssh/authorized_keys, то не будет работать проверка задач (ключ используется для проверки), однако достаточно просто пересоздать машину и ключ снова появится.

Теперь можно зайти на ВМ по созданному ключу с rds
```shell
ssh user@st{номер} -i myid_rsa
---
Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-101-generic x86_64)
...
```

Если пару `myid_rsa` и `myid_rsa.pub` переместить в папку `~/.ssh`, то заходить по ним можно без использования опции `-i` (ssh автоматически пробует все ключи из этой папки при подключении к хосту). Можно сгенерировать пару ключей просто командой `ssh-keygen` без опции `-f`. Тогда сгенерированная пара ключей пропишется в папке ~/.ssh

