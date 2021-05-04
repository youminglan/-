## MySQL基本语法



### 对表的操作

1. 增加表：

   ```sql
   CREATE TABLE 表名 (	列名	类型	)
   ```

2. 修改表：

    ```sql
     ALTER TABLE 表名
     	ADD (列名	数据类型);
    ```

    ```sql
     ALTER TABLE 表名
     	MODIFY (列名	数据类型);
    ```

3. 查看表

   ```sql
   SHOW TABLES
   ```
   
   ```sql
   SHOW CREATE TABLE 表名【查看表的创建细节】
   ```
   
   ```sql
   DESC 表名【查看表的结构】
   ```
   
4. 删除表

    ```sql
    ALTER TABLE 表名
    DROP(列名);
    ```

    

### 对表中数据的操作

1. 增加

   ```sql
   INSERT INTO 表名 (	列名	)
   VALUES	(数据..);
   ```

2. 修改

   ```sql
   UPDATE 表名
   SET 列名=值..,列名=值
   WHERE=条件
   ```

3. 删除

   ```sql
   DELETE FROM 表名 WHERE=条件
   ```

   ``` sql
   TRUNCATE TABLE【先摧毁整张表，再创建表结构】
   ```

4. 查看

   ```sql
   SELECT 列名
   FROM 表名,
   WHERE 条件,
   GROUP BY 列名,
   HAVING BY,
   ORDER BY 列名
   ```

   ### 连接数据库

   ```sql
   mysql -u root -p
   ```

   ### 对库的操作

   1. 创建库

      ```sql
      CREATE DATABASE [IF NOT EXISTS] 库名
      [DEFAULT] CHARACTER SET 字符名 | [DEFAULT] COLLATE 校对规则
      ```

   2. 查看库

      ```sql
      SHOW DATABASES
      ```

      ```sql
      SHOW CREATE DATABASE 库名【查看数据库创建时的详细信息】
      ```

   3. 删除库

      ```sql
      DROP DATABASE [IF EXISTS] 库名
      ```

   4. 修改库

      ```sql
      ALTER DATABASE [IF NOT EXISTS] 库名
      [DEFAULT] CHARACTER SET 字符名 | [DEFAULT] COLLATE 校对规则
      ```

   5. 备份库中的数据和

      ```sql
      mysqldump -u 用户名 -p 数据库名 > 文件名.sql[WINDOWS命令]
      ```

      ```sql
      Source 文件名.sql[在库下执行]
      ```

      ```sql
      mysql -uroot -p mydbl<c:test.sql [WINDOWS命令]
      ```

      