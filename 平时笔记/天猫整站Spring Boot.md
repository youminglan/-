---
title: hexo的个性化配置
date: 
author: youminglan
img: 
top: true
cover: true
coverImg: 
password: 
toc: false
mathjax: false
summary: hexo的matery主题的个性化配置
categories: hexo
tags:
  - hexo

---

# 使用Spring Boot搭建一个完整的天猫商城



## 基础

### 前台

####  项目介绍


1. 首页
2. 分类页
3. 查询结果页
4. 产品页
5. 结算页面
6. 支付页面
7. 支付成功页面
8. 购物车页面
9. 我的订单页面
10. 确认收货页面
11. 确认收货成功页面
12. 进行评价页面
13. 登录页面
14. 注册页面



### 后台

#### 项目介绍

1. 分类管理
2. 分类属性管理
3. 产品管理
4. 产品属性设置
5. 产品图片管理
6. 用户管理
7. 订单管理



#### 开发流程

1. 技术准备

为了完成这个J2ee项目，需要掌握如下技术，当然也可以以本项目为驱动，**哪里不懂，学哪里**，其实这也是最好的学习方式(带着目标去学习)
Java
[Java基础](https://how2j.cn/stage/12.html) 和 [Java中级](https://how2j.cn/stage/25.html) 的大部分内容

前端
[html](https://how2j.cn/k/html/html-tutorial/175.html), [CSS](https://how2j.cn/k/css2/css2-tutorial/238.html), [Javascript](https://how2j.cn/k/javascript/javascript-javascript-tutorial/519.html), [JSON](https://how2j.cn/k/json/json-tutorial/531.html), [AJAX](https://how2j.cn/k/ajax/ajax-tutorial/465.html), [JQuery ](https://how2j.cn/k/jquery/jquery-tutorial/467.html),[Bootstrap](https://how2j.cn/k/boostrap/boostrap-tutorial/538.html), [Vue.js](https://how2j.cn/k/vuejs/vuejs-start/1744.html)

框架部分
[spring](https://how2j.cn/k/spring/spring-ioc-di/87.html) [springmvc](https://how2j.cn/k/springmvc/springmvc-springmvc/615.html) [springboot](https://how2j.cn/k/springboot/springboot-eclipse/1640.html)

中间件
[redis](https://how2j.cn/k/redis/redis-download-install/1367.html), [nginx](https://how2j.cn/k/nginx/nginx-tutorial/1565.html), [elasticsearch](https://how2j.cn/k/search-engine/search-engine-start/1691.html), [shiro](https://how2j.cn/k/shiro/shiro-plan/1732.html)

数据库：
[MySQL](https://how2j.cn/k/mysql/mysql-install/377.html)

开发工具
[Intellij IDEA](https://how2j.cn/k/idea/idea-download-install/1348.html),[Maven](https://how2j.cn/k/maven/maven-introduction/1328.html)

2. 技术准备

   1. 需求分析

      首先确定要做哪些功能，需求分析包括[前台](https://how2j.cn/k/tmall_springboot/tmall_springboot-1812/1812.html)和[后台](https://how2j.cn/k/tmall_springboot/tmall_springboot-1813/1813.html)。
      前台又分为单纯要展示的那些功能-[需求分析-展示](https://how2j.cn/k/tmall_springboot/tmall_springboot-1812/1812.html)，以及会提交数据到服务端的哪些功能-[需求分析-交互](https://how2j.cn/k/tmall_springboot/tmall_springboot-1867/1867.html)。

   2. 表结构设计

      接着是表结构设计，表结构设计是围绕功能需求进行，如果表结构设计有问题，那么将会影响功能的实现。除了[表与表关系](https://how2j.cn/k/tmall_springboot/tmall_springboot-1825/1825.html)，[建表SQL语句](https://how2j.cn/k/tmall_springboot/tmall_springboot-1824/1824.html)之外，为了更好的帮助大家理解表结构以及关系，还特意把[表与页面功能](https://how2j.cn/k/tmall_springboot/tmall_springboot-1822/1822.html)一一对应起来

   3. 原型

      接着是界面原型，与客户沟通顺畅的项目设计流程里一定会有原型这个环节。 借助界面原型，可以**低成本，高效率**的与客户达成需求的一致性。 同样的，原型分为了[前台原型](https://how2j.cn/k/tmall_springboot/tmall_springboot-1841/1841.html)和[后台原型](https://how2j.cn/k/tmall_springboot/tmall_springboot-1842/1842.html)。

   4. 后台-分类管理

      接下来开始进行功能开发，按照模块之间的依赖关系，首先进行[后台-分类管理](https://how2j.cn/k/tmall_springboot/tmall_springboot-1893/1893.html)功能开发。严格来说，这是开发的第一个功能，所以讲解的十分详细，不仅提供了[可运行的项目](https://how2j.cn/k/tmall_springboot/tmall_springboot-1892/1892.html)，还详细解释了其中用到的[HTML 包含关系](https://how2j.cn/k/tmall_springboot/tmall_springboot-1884/1884.html)，以及每个具体的功能： [查询](https://how2j.cn/k/tmall_springboot/tmall_springboot-1891/1891.html)，[分页](https://how2j.cn/k/tmall_springboot/tmall_springboot-1890/1890.html)，[增加](https://how2j.cn/k/tmall_springboot/tmall_springboot-1889/1889.html)，[删除](https://how2j.cn/k/tmall_springboot/tmall_springboot-1888/1888.html)，[编辑](https://how2j.cn/k/tmall_springboot/tmall_springboot-1887/1887.html)，[修改](https://how2j.cn/k/tmall_springboot/tmall_springboot-1886/1886.html)。 把每个细节都掰的很细，可以更好的理解，消化和吸收。 在把[后台-分类管理](https://how2j.cn/k/tmall_springboot/tmall_springboot-1893/1893.html) 吃透之后，后续的其他后台管理功能，做起来就会更加顺畅。

   5. 后台-其他管理

      在把[后台-分类管理](https://how2j.cn/k/tmall_springboot/tmall_springboot-1893/1893.html) 消化吸收之后，就可以加速进行 [后台其他页面](https://how2j.cn/k/tmall_springboot/tmall_springboot-1856/1856.html)的学习。

   6. 前台-首页

      前台也包括许多功能， 与[后台-分类管理](https://how2j.cn/k/tmall_springboot/tmall_springboot-1893/1893.html)类似的，首先把[前台-首页](https://how2j.cn/k/tmall_springboot/tmall_springboot-1863/1863.html)这个功能单独拿出来，进行精讲。[前台-首页](https://how2j.cn/k/tmall_springboot/tmall_springboot-1863/1863.html) 消化吸收好之后，再进行其他前台功能的开发。

   7. 前台无需登录 

      从前台模块之间的依赖性，以及开发顺序的合理性来考虑，把前台功能分为了 [无需登录](https://how2j.cn/k/tmall_springboot/tmall_springboot-1901/1901.html) 即可使用的功能，和[需要登录](https://how2j.cn/k/tmall_springboot/tmall_springboot-1914/1914.html) 才能访问的功能。 建立在前一步[前台-首页](https://how2j.cn/k/tmall_springboot/tmall_springboot-1863/1863.html)的基础之上，开始进行一系列的[无需登录](https://how2j.cn/k/tmall_springboot/tmall_springboot-1901/1901.html)功能开发。

   8. 前台需要登录 

      最后是[需要登录的前台功能](https://how2j.cn/k/tmall_springboot/tmall_springboot-1914/1914.html)。 这部分功能基本上都是和购物相关的。 因此，一开始先把[购物流程](https://how2j.cn/k/tmall_springboot/tmall_springboot-1914/1914.html) 单独拿出来捋清楚，其中还特别注明了[购物流程环节与表关系](https://how2j.cn/k/tmall_springboot/tmall_springboot-1914/1914.html#step8612)，这样能够更好的建立对前端购物功能的理解。随着这部分功能的开发，就会进入订单生成部分，在此之前，先准备了一个 [订单状态图](https://how2j.cn/k/tmall_springboot/tmall_springboot-1903/1903.html#step8539)，在理解了这个图之后，可以更好的进行订单相关功能的开发。

   9. 总结

      最后总结整个项目的项目结构，都实现了哪些典型场景，运用了哪些设计模式，把学习到的知识都沉淀下来，转换，消化，吸收为自己的技能

      

      

      ## 需求分析

      ### 展示

      1. 前端展示

         在前端页面上显示数据库中的数据，如首页，产品页，购物车，分类页面等等。
   至于这些前端页面如何组织显示，页面布局，css样式设置，Javascript交互代码等教学，在单独的模仿天猫前端教程中详细讲解。
         分开学习和讲解，降低学习的难度，避免全部前后端混杂在一起学带来的困扰。

      2. 前端交互

         这里的前端交互，与模仿天猫前端教程里的交互，不是同一个概念。 模仿天猫前端教程里的交互，仅仅停留在浏览器上的javascript交互，这里的交互指的是通过POST,GET等http协议，与服务端进行同步或者异步数据交互。 比如购买，购物车，生成订单，登录等等功能。

      3. 后台功能

         对支撑整站需要用到的数据，进行管理维护。 比如分类管理，分类属性管理， 产品管理，产品图片管理，用户管理，订单管理等等
      
      
      
      ### 交互
      
      交互不是仅仅停留在浏览器上的JavaScript交互，这里的交互指的是通过POST，等http协议，与服务端进行同步或者异步数据交互。 比如购买，购物车，生成订单，登录等等功能。
      
      1. 分类页排序
      2. 立即购买
      3. 加入购物车
      4. 调整订单数量
      5. 删除订单页数量
      6. 删除订单页
      7. 生成订单
      8. 订单页功能
      9. 确认付款
      10. 确认收货
      11. 提交评价信息
      12. 登录
      13. 注册
      14. 退出
      15. 搜索
      
      
      
      #### 前端页面需求列表清单
      
      1. 首页
         - 在横向导航栏上提供4个分类连接
         - 在纵向导航栏上提供全部17个分类连接
         - 当鼠标移动到某一个纵向分类连接的时候，显示这个分类下的推荐商品
         - 按照每种分类，显示5个商品的方式显示所有17种分类
      2. 产品页
         - 显示分辨率为950x100的当前商品对应的分类图片
         - 显示本商品的5个单独图片
         - 商品的基本信息，如标题，小标题，加个，销量，评价数量，库存等
         - 商品详情
         - 评价信息
         - 5张商品详细图片
         - 立即购买
         - 加入购物车
      3. 分类页
         - 显示分辨率为950x100的当前分类图片
         - 显示本分类下的所有产品
         - 分类页排序
      4. 搜索结果页
         - 显示满足查询条件的商品
      5. 购物车查看页
         - 在购物车显示订单项
      6. 结算页
         - 在结算页面显示被选中的订单项
         - 生成订单
      7. 确认订单页
         - 确认支付页面显示本次订单的金额总数
         - 确认付款
      8. 支付成功页
         - 付款成功时，显示本次付款金额
      9. 我的订单页
         - 显示所有订单，以及对应的订单项
      10. 确认收货页
          - 显示订单项内容
          - 显示订单信息，收货人地址等
          - 确认收货
      11. 评价页
          - 显示要评价的商品信息，商品当前的总评价数
          - 评价成功后，显示当前商品所有的评价信息
          - 提交评价信息
      12. 页头信息展示
      
      1. - 未登录状态
         - 已登录状态
         - 登录
         - 注册
         - 退出
      
      13. 所有页面
          - 搜索
      
      
      
      
      
      
      
      
      
      
      
      
      
      

