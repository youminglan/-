//给你两个二进制字符串，返回它们的和（用二进制表示）
//输入为非空字符串且只包含数字 1 和 0。

//输入11，1返回 110
package Leetcode;

public class Leetcode_67 {

    public String addBinary(String a, String b){
        if(a == null || a.length() == 0) return b;
        if(b == null || b.length() == 0) return a;

        //拼接字符串
        StringBuilder stb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;

        // 进位

        int c = 0;
        while(i >= 0 || j>=0){
            if(i>=0) c += a.charAt(i--) - '0';
            if(j>=0) c += b.charAt(j--) - '0';
            stb.append(c%2);
            c >>= 1;
        }

        String res = stb.reverse().toString();
        return  c > 0 ? '1' + res : res;

    }
    public static void main(String[] args) {

    }
}
