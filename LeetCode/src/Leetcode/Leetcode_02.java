package Leetcode;
import Leetcode.ListNode;

//请判断一个链表是否为回文链表。

public class Leetcode_02 {
    public boolean isPalindrome(ListNode head) {
        ListNode fast = head,slow = head;
        //通过指针快慢找到中点
        while (fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        //如果fast不为空，则说明链表长度是奇数
        if (fast != null){
            slow = slow.next;
        }
        slow = reverse(slow);

        fast = head;
        while (slow != null)
        {
            if(fast.val != slow.val)
                return false;
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }
    public ListNode reverse(ListNode head){
        ListNode prev = null;
        while (head != null){
            ListNode next = head.next;
            head.next = prev;
            head = next;
        }
        return prev;
    }

    public static void main(String[] args) {

    }
}
