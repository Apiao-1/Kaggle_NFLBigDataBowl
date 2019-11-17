# EDA

### Data

https://www.kaggle.com/c/nfl-big-data-bowl-2020/data

- `GameId` - a unique game identifier 比赛ID
- `PlayId` - a unique play identifier 一场比赛中所有play的ID（每次play都有可能得分）
- `Team` - home or away 队名
- `X` - player position along the long axis of the field. See figure below.
- `Y` - player position along the short axis of the field. See figure below.
 ![img](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3258%2F7542d363a19fa3eea77708e6b90bc420%2FFig1.png?generation=1570562067917019&alt=media)
> NFL和NCAA使用的标准球场是一个长360英尺（120码或109.7米）、宽160英尺（53.33码或48.8米）的长方形草坪（有些室内赛会使用仿草地毯），较长的边界称为边线（sideline），较短的边界称为端线（end line）。端线前的标示线称为得分线（goal line），球场每侧端线与得分线之间有一个纵深10码（9.1米）的得分区叫做端区（end zone，也称达阵区），端区的四角各有一个约有1英尺长的橙色长方体标柱（pylon）。两侧得分线相距100码（91.44米），之间的区域也就是比赛区（playing field）。比赛区上距离得分线每5码（4.6米）距离标划一条码线（yard line，或称5码线），每10码标示数字，直到50码线到达中场（midfield）。在球场中间和两侧与边线平行排列划有横向的短标示线，称为码标（hash marks，或称整码线），其中接近边线的码标线称为界内线（inbounds line）。任何球员都必须在码标线上或之间进行发球。
- `S` - speed in yards/second 此时的速度

- `A` - acceleration in yards/second^2 加速度

- `Dis` - distance traveled from prior time point, in yards

- `Orientation` - orientation of player (deg) 玩家面对的方向

- `Dir` - angle of player motion (deg) 玩家移动的方向

- ![img](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F75976%2F277b02ac1a245d56362715d8a550fb74%2Forientation.png?generation=1571665174994396&alt=media)

- `NflId` - a unique identifier of the player 运动员ID

- `DisplayName` - player's name 运动员name

- `JerseyNumber` - jersey number 运动员号码

- `Season` - year of the season

- `YardLine` - the yard line of the line of scrimmage 发球的码线

- `Quarter` - game quarter (1-5, 5 == overtime) 比赛所处的时间

- `GameClock` - time on the game clock 

- `PossessionTeam` - team with possession 当前拥有控球权的队伍

- `Down` - the down (1-4)

  > 进攻方有四次机会向前方（防守方的端区）累计推进10码，每次机会称为一“**档**” 进攻（down，即被对方拦截放倒一次的机会）。当进攻方成功的在四档进攻内推进了10码以上，便可获得新的四档进攻机会——称为获得新的 “**首档**”（1st down，也称**首攻**）。通过不断获得新的首攻，进攻方可以进行连续的系列进攻向前不断推进，直至得分。而防守方的目的也很简单——就是尽可能阻止对方在四档进攻内推进足够的距离，逼迫其交换控球权。

- `Distance` - yards needed for a first down 距离新的首攻所需要的码数

- `FieldPosition` - which side of the field the play is happening on play发生在哪个球队的半场

- `HomeScoreBeforePlay` - home team score before play started 主队已经获得的比分

- `VisitorScoreBeforePlay` - visitor team score before play started 客队已经获得的比分

- `NflIdRusher` - the `NflId` of the rushing player 进攻方持球选手ID

- `OffenseFormation` - offense formation

- `OffensePersonnel` - offensive team positional grouping 进攻队员

- `DefendersInTheBox` - number of defenders lined up near the line of scrimmage, spanning the width of the offensive line

- `DefensePersonnel` - defensive team positional grouping 防守队员

- `PlayDirection` - direction the play is headed

- `TimeHandoff` - UTC time of the  handoff 传球时间

- `TimeSnap` - UTC time of the snap 发球的时间

- **`Yards` - the yardage gained on the play (you are predicting this)** 得分

- `PlayerHeight` - player height (ft-in)

- `PlayerWeight` - player weight (lbs)

- `PlayerBirthDate` - birth date (mm/dd/yyyy)

- `PlayerCollegeName` - where the player attended college

- `Position` - the player's position (the specific role on the field that they typically play)

- `HomeTeamAbbr` - home team abbreviation 主队缩写

- `VisitorTeamAbbr` - visitor team abbreviation

- `Week` - week into the season 赛季的第几周

- `Stadium` - stadium where the game is being played

- `Location` - city where the game is being player

- `StadiumType` - description of the stadium environment 体育馆类型

- `Turf` - description of the field surface 场地类型

- `GameWeather` - description of the game weather

- `Temperature` - temperature (deg F)

- `Humidity` - humidity 湿度

- `WindSpeed` - wind speed in miles/hour

- `WindDirection` - wind direction

# Question
1. play的特征和每个球员的特征如何统一进模型中
    2. NN可以对不同size 的特征进行处理
    1. play的特征中加入rusher的特征作为球员特征
    3. play的特征中加入每一个球员的所有特征，不建议（球员特征多的时候*22 引入大量噪声）
2. 多分类问题如何形成评价指标
    1. 连续分级概率评分（Continuous Ranked Probability Score, CRPS），按CRPS评价概率模型所得的（优劣）结果与按MAE评价概率模型的数学期望所得的结果等价，train model时用mae
    2. https://baike.baidu.com/item/%E8%BF%9E%E7%BB%AD%E5%88%86%E7%BA%A7%E6%A6%82%E7%8E%87%E8%AF%84%E5%88%86/23690953?fr=aladdin
3. play的数据量只有23171 * 72，会造成过拟合
4. sklearn中的MAE是负值，原因：因为有些score是越大越好，比如roc_auc,但有些越小越好比如各种loss，为了统一，sklearn为最小化的值加了负号转化为最大化的问题，这里需要相应地修改网格里的初始化参数
5. 关于运动员的运动方向，进攻方可能向左可能向右，此时需要对yard进行转换，处理时有个trick是将X,y,角度都进行翻转，保证进攻方的方向始终是一致的

kaggle的弊端在于工程化毫无提现，代码习惯太差了，强行要揉到一个文件里面，上千行代码，纯面向过程