//给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。如果不存在最后一个单词，请返回 0 。
//
//        单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。


package Leetcode;

public class Leetcode_58 {

    public int lengthOfLastWord(String s) {
        int length = 0;
        for (int i  = s.length() - 1;i>=0;i--){
            if(s.charAt(i)!=' '){
                length++;

            }
            else if (length != 0)
                return length;
        }
        return length;
    }

    public static void main(String[] args) {

    }


}
