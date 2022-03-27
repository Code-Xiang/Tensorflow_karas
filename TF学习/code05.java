public class code05{
    // 摆法符合题目意思：排完序之后，如果数字分布一样，就认为是同一种摆法！
    // 给你apples，给你plates
    // 返回几种摆法
     public static int f(int apples, int plates){
           if(apples == 0){
               return 1;
           }
           // apples > 1
           if(plates == 0){
               return 0; 
           }
           if (apples < plates){ // 5  100
               return f(apples, apples);
           }else { // apples > plates
            // 1) 每个盘子先来一个苹果，剩下apples - plates，继续往plates盘子里放
            // 2) 盘子不全用，砸掉一个，f(apples,plates-1)
               return f(apples-plates,plates) + f(apples,plates-1);
           }
     }
}