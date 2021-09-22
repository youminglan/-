package Leetcode;

//在有序数组中找出两个数，使它们的和为 target
/**
 * @author : youminglan
 * @date : 2021/9/22 10:06
 * Create in Wuhan Hubei
 */
public class Leetcode_167 {
    public int[] twoSum(int[] numbers,int target){
        int i = 0,j = numbers.length-1;
        while(i<j){
            int sum = numbers[i] + numbers[j];
            if(sum == target)
                return new int[]{i+1,j+1};
            else if(sum < target)
                i++;
            else
                j--;
        }
        return null;
    }
    public static void main(String[] args) {

    }
}
