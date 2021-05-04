## 服务端面试题 ## 



- 9种隐式对象，以及他们的用途

  JSP一共有9个隐式对象，分别是
  **request,response,out**
  分别代表请求，响应和输出

  **pageContext, session,application**
  pageContext 代表当前页面作用域
  session 代表当会话作用域
  application 代表当全局作用域

  **config**
  config可以获取一些在web.xml中初始化的参数.

  **page**
  表示当前对象

  **exception**
  表示异常对象.



- 3种JSP的指令

  [<%@page](https://how2j.cn/k/jsp/jsp-tutorials/530.html#step1651)
  JSP的基本设置，比如编码方式，import其他类，是否开启EL表达式

  [<%@include](https://how2j.cn/k/jsp/jsp-include/576.html)
  包含其他的文件

  [<%@taglib](https://how2j.cn/k/jsp/jsp-jstl/578.html)
  使用标签库



- 2种JSP的动作

  <jsp:forword
  服务端跳转

  <jsp:include
  包含其他文件



- doGet()和 doPost的区别，分别在什么情况下调用

  doGet和doPost都是在[service()](https://how2j.cn/k/servlet/servlet-service/549.html)方法后调用的，分别来处理method="get"和method="post"的请求



- servlet的init方法和service方法的区别

  在Servlet的生命周期中，先调用init进行初始化，而且只调用一次。

  接着再调用service,有多少次请求，就调用多少次service



- servlet的生命周期

  一个Servlet的生命周期由 [实例化](https://how2j.cn/k/servlet/servlet-lifecycle/550.html#step1594)，[初始化](https://how2j.cn/k/servlet/servlet-lifecycle/550.html#step1595)，[提供服务](https://how2j.cn/k/servlet/servlet-lifecycle/550.html#step1596)，[销毁](https://how2j.cn/k/servlet/servlet-lifecycle/550.html#step1597)，[被回收](https://how2j.cn/k/servlet/servlet-lifecycle/550.html#step1597) 几个步骤组成

  ![servlet的生命周期](https://stepimagewm.how2j.cn/1735.png)



- 页面间对象传递的方法

  假设是a.jsp传递数据到b.jsp，那么页面间对象传递的方式有如下几种

  1. 在a.jsp中request.setAttribute，然后**服务端跳转**到b.jsp

  2. 在a.jsp中session.setAttribute，然后跳转到b.jsp, 无所谓客户端还是服务端跳转

  3. 在a.jsp中application.setAttribute, 然后跳转到b.jsp，无所谓客户端还是服务端跳转



- Request常见方法

  **request.getRequestURL():** 浏览器发出请求时的完整URL，包括协议 主机名 端口(如果有)" +

  **request.getRequestURI():** 浏览器发出请求的资源名部分，去掉了协议和主机名" +

  **request.getQueryString():** 请求行中的参数部分，只能显示以get方式发出的参数，post方式的看不到

  **request.getRemoteAddr():** 浏览器所处于的客户机的IP地址

  **request.getRemoteHost():** 浏览器所处于的客户机的主机名

  **request.getRemotePort():** 浏览器所处于的客户机使用的网络端口

  **request.getLocalAddr():** 服务器的IP地址

  **request.getLocalName():** 服务器的主机名

  **request.getMethod():** 得到客户机请求方式一般是GET或者POST



- J2EE是技术，还是平台，还是框架

  是**平台**，上面运行各种各样的技术(servlet,jsp,filter,listner)和框架(struts,hibernate,spring)



- 编写JavaBean的注意事项

  JavaBean就是实体类
  无参构造方法
  属性都用private修饰，并且都有public的getter和setter



- MVC的各个部分都有哪些技术来实现，分别如何实现

  M 模型层代表数据，使用bean,dao等等
  V 视图层代表展现，使用html,jsp,css
  C 控制层代表控制，使用servlet



- JSP中两种include的区别

  一种是静态包含，一种是动态包含



- 简述你对简单Servlet、过滤器、监听器的理解

  Servlet的作用是处理获取参数，处理业务，页面跳转

  过滤器（filter）的作用是拦截请求，一般会有做编码处理，登录权限验证

  监听器（listner）的作用是监听Request，Session，Context等等的生命周期，以及其中数据的变化