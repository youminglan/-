//假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
//
//每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
//
//注意：给定 n 是一个正整数。

package Leetcode;

public class Leetcode_70 {

    //动态规划法


    public int climbStairs(int n) {
        int [] step = new int[n+1];
        step[0] = 1;
        step[1] = 1;
        for (int i = 2;i <= n;i++){
            step[i] = step[i-1] + step[i-2];
        }
        return step[n];
    }

    public static void main(String[] args) {

    }
}
